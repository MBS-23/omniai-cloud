"""HTTP surface: validation, limits, errors and streaming."""

from __future__ import annotations

import io
import json

import pytest


def _upload(client, filename: str, payload: bytes, endpoint: str = "/api/analyze/file"):
    return client.post(
        endpoint,
        data={"file": (io.BytesIO(payload), filename)},
        content_type="multipart/form-data",
    )


class TestHealth:
    def test_reports_profile_and_capabilities(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "ok"
        assert body["profile"] in {"lite", "paper"}
        assert "classify_image" in body["capabilities"]

    def test_never_leaks_the_api_key(self, client):
        body = client.get("/api/health").get_data(as_text=True)
        assert "api_key" not in body.lower()
        assert "test-key-not-used" not in body


class TestUploadValidation:
    def test_missing_file_is_rejected(self, client):
        response = client.post("/api/analyze/file", data={})
        assert response.status_code == 400
        assert "Select a file" in response.get_json()["detail"]

    @pytest.mark.parametrize("filename", ["payload.exe", "archive.zip", "notes.txt"])
    def test_disallowed_extensions_rejected(self, client, filename):
        response = _upload(client, filename, b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        assert response.status_code == 400
        assert "Unsupported file type" in response.get_json()["detail"]

    def test_renamed_file_is_caught_by_magic_bytes(self, client):
        """A Windows executable renamed to .png must not reach a model."""
        response = _upload(client, "innocent.png", b"MZ\x90\x00" + b"\x00" * 128)
        assert response.status_code == 400
        assert "does not look like a valid" in response.get_json()["detail"]

    def test_empty_file_is_rejected(self, client):
        response = _upload(client, "empty.png", b"")
        assert response.status_code == 400
        assert "empty" in response.get_json()["detail"].lower()

    def test_decompression_bomb_is_rejected(self, client, huge_image):
        response = _upload(client, "bomb.png", huge_image.read_bytes())
        assert response.status_code == 400
        detail = response.get_json()["detail"]
        assert "too large to process" in detail
        assert "megapixel" in detail

    def test_valid_image_is_accepted(self, client, scene_image):
        response = _upload(client, "scene.png", scene_image.read_bytes())
        assert response.status_code == 200
        body = response.get_json()
        assert "metadata" in body and "results" in body and "explanation" in body


class TestTextValidation:
    def test_empty_text_rejected(self, client):
        response = client.post("/api/analyze/text", data={"text": "   "})
        assert response.status_code == 400

    def test_oversized_text_rejected(self, client, settings):
        oversized = "a " * (settings.limits.max_text_chars)
        response = client.post("/api/analyze/text", data={"text": oversized})
        assert response.status_code == 400
        assert "too long" in response.get_json()["detail"]

    def test_accepts_json_body(self, client):
        response = client.post(
            "/api/analyze/text",
            data=json.dumps({"text": "Analyse this English sentence.", "target_language": "en"}),
            content_type="application/json",
        )
        assert response.status_code == 200


class TestRateLimiting:
    def test_limit_is_enforced_and_advertises_retry_after(self, app, client):
        from backend.utils.security import SlidingWindowRateLimiter

        app.extensions["omniai_rate_limiter"] = SlidingWindowRateLimiter(2)

        assert client.post("/api/analyze/text", data={"text": "one two three"}).status_code == 200
        assert client.post("/api/analyze/text", data={"text": "one two three"}).status_code == 200

        blocked = client.post("/api/analyze/text", data={"text": "one two three"})
        assert blocked.status_code == 429
        assert blocked.get_json()["retry_after"] >= 1


class TestSecurityHeaders:
    def test_headers_present_on_every_response(self, client):
        headers = client.get("/api/health").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]

    def test_only_huggingface_may_frame_the_app_on_a_space(self, monkeypatch, backend):
        """The Space page embeds the app; no other site may."""
        from backend.app import create_app
        from backend.config import reload_settings
        from backend.providers import reset_backend, set_backend

        monkeypatch.setenv("SPACE_ID", "NanicyberAI/omniai-cloud")
        application = create_app(reload_settings(), warm=False)
        set_backend(backend)
        try:
            headers = application.test_client().get("/api/health").headers
        finally:
            reset_backend()
        assert "frame-ancestors 'self' https://huggingface.co;" in headers["Content-Security-Policy"]
        assert "X-Frame-Options" not in headers

    def test_csp_allows_no_third_party_scripts(self, client):
        csp = client.get("/api/health").headers["Content-Security-Policy"]
        assert "script-src 'self'" in csp
        assert "cdn.jsdelivr.net" not in csp


class TestStreaming:
    def _events(self, response) -> list[tuple[str, dict]]:
        parsed = []
        for block in response.get_data(as_text=True).split("\n\n"):
            if not block.strip():
                continue
            name = value = None
            for line in block.splitlines():
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: "):
                    value = json.loads(line[6:])
            if name:
                parsed.append((name, value))
        return parsed

    def test_text_stream_emits_progress_then_result(self, client):
        response = client.post(
            "/api/analyze/text/stream",
            data={"text": "Analyse this English sentence please.", "target_language": "en"},
        )
        assert response.status_code == 200
        assert response.mimetype == "text/event-stream"

        events = self._events(response)
        names = [name for name, _ in events]
        assert "progress" in names
        assert names[-1] == "result"

        final = events[-1][1]
        assert "selected_modules" in final and "explanation" in final

    def test_file_stream_reports_real_stage_names(self, client, scene_image):
        response = _upload(client, "scene.png", scene_image.read_bytes(),
                           endpoint="/api/analyze/file/stream")
        events = self._events(response)
        stages = {payload["stage"] for name, payload in events if name == "progress"}
        # These are pipeline steps, not a timer's invented labels.
        assert {"inspect", "route", "image_classification"} <= stages

    def test_stream_sets_no_hop_by_hop_headers(self, client):
        """PEP 3333 forbids them, and real WSGI servers return 500 if set.

        Flask's development server tolerates a Connection header, so this
        regression is invisible until deployment. waitress and gunicorn both
        raise AssertionError on it.
        """
        response = client.post(
            "/api/analyze/text/stream",
            data={"text": "Analyse this English sentence please.", "target_language": "en"},
        )
        hop_by_hop = {
            "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
            "te", "trailer", "transfer-encoding", "upgrade",
        }
        # Werkzeug's Headers iterates (key, value) pairs, so .keys() is needed.
        offending = [k for k in response.headers.keys() if k.lower() in hop_by_hop]  # noqa: SIM118
        assert not offending, f"hop-by-hop header(s) set by the app: {offending}"

    def test_stream_reports_failure_as_an_event(self, client, backend, scene_image):
        backend.unavailable = {
            "classify_image", "detect_objects", "extract_text", "caption_image",
        }
        response = _upload(client, "scene.png", scene_image.read_bytes(),
                           endpoint="/api/analyze/file/stream")
        events = self._events(response)
        assert any(name == "result" for name, _ in events)
        final = [p for n, p in events if n == "result"][0]
        assert final["warnings"], "degraded capabilities must be reported"


class TestSecretRedaction:
    """No credential may reach a log, an error body or the browser."""

    def test_redacts_common_secret_shapes(self):
        from backend.utils.redaction import redact

        cases = [
            ("Authorization: Bearer sk-abc123def456ghi789", "sk-abc123def456ghi789"),
            ('{"api_key": "mySuperSecretValue"}', "mySuperSecretValue"),
            ("token=ghp_FAKEtokenFORtestingONLYnotREAL0000", "ghp_FAKEtokenFORtestingONLYnotREAL0000"),
            ("password: hunter2hunter2", "hunter2hunter2"),
        ]
        for text, secret in cases:
            cleaned = redact(text)
            assert secret not in cleaned, f"{secret!r} survived redaction of {text!r}"
            assert "[redacted]" in cleaned

    def test_redacts_the_configured_key_verbatim(self, settings, monkeypatch):
        from backend.utils import redaction

        monkeypatch.setattr(redaction, "_literal_secrets", lambda: ["test-key-not-used"])
        cleaned = redaction.redact("upstream said: test-key-not-used is invalid")
        assert "test-key-not-used" not in cleaned

    def test_leaves_ordinary_prose_untouched(self):
        from backend.utils.redaction import redact

        message = "Translation did not run: the model was unavailable."
        assert redact(message) == message

    def test_error_responses_are_redacted(self, client, backend, monkeypatch):
        from backend.providers.base import BackendUnavailable
        from backend.utils import redaction

        monkeypatch.setattr(redaction, "_literal_secrets", lambda: ["supersecretkey123"])

        def boom(*args, **kwargs):
            raise BackendUnavailable("upstream rejected key supersecretkey123")

        monkeypatch.setattr(backend, "analyze_text", boom)
        response = client.post("/api/analyze/text", data={"text": "Analyse this sentence."})
        body = response.get_data(as_text=True)
        assert "supersecretkey123" not in body
