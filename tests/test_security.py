"""Protections at the HTTP boundary, checked the way an attacker would probe them."""

from __future__ import annotations

import zipfile

import pytest

from backend.utils import docx_utils
from backend.utils.docx_utils import validate_docx_file
from backend.utils.security import (
    AnalysisSlots,
    UploadRejected,
    client_identifier,
    validate_image_dimensions,
)

_TEXT = {"text": "The results were excellent and everyone was pleased."}


class TestClientIdentity:
    """Rate limits key on the caller; the caller must not be able to choose its key."""

    def test_forwarded_header_is_ignored_without_a_trusted_proxy(self):
        assert client_identifier("10.0.0.5", "1.2.3.4") == "10.0.0.5"

    def test_behind_one_proxy_the_rightmost_entry_is_the_caller(self):
        # The client wrote the first entry itself; the proxy appended the second.
        assert client_identifier("10.0.0.1", "6.6.6.6, 203.0.113.9", trusted_proxies=1) == "203.0.113.9"

    def test_invented_entries_do_not_create_new_identities(self):
        identities = {
            client_identifier("10.0.0.1", f"{n}.{n}.{n}.{n}, 203.0.113.9", trusted_proxies=1)
            for n in range(1, 60)
        }
        assert identities == {"203.0.113.9"}

    def test_a_short_header_falls_back_to_the_socket_address(self):
        assert client_identifier("10.0.0.1", "203.0.113.9", trusted_proxies=2) == "10.0.0.1"


class TestCrossSiteRequests:
    def test_a_request_from_another_site_is_refused(self, client):
        response = client.post("/api/analyze/text", json=_TEXT, headers={"Sec-Fetch-Site": "cross-site"})
        assert response.status_code == 403

    def test_a_foreign_origin_is_refused(self, client):
        response = client.post("/api/analyze/text", json=_TEXT, headers={"Origin": "https://attacker.example"})
        assert response.status_code == 403

    def test_the_sites_own_page_is_accepted(self, client):
        response = client.post(
            "/api/analyze/text",
            json=_TEXT,
            headers={"Origin": "http://localhost", "Sec-Fetch-Site": "same-origin"},
        )
        assert response.status_code == 200

    def test_a_script_with_no_browser_headers_is_accepted(self, client):
        assert client.post("/api/analyze/text", json=_TEXT).status_code == 200


class TestBusyServer:
    def test_a_full_server_asks_the_caller_to_retry(self, app, client):
        slots = AnalysisSlots(1, wait_seconds=0.01)
        app.extensions["omniai_analysis_slots"] = slots
        slots.acquire()
        try:
            response = client.post("/api/analyze/text", json=_TEXT)
        finally:
            slots.release()
        assert response.status_code == 503
        assert response.get_json()["retry_after"] > 0

    def test_every_analysis_gives_its_slot_back(self, app, client):
        app.extensions["omniai_analysis_slots"] = AnalysisSlots(1, wait_seconds=0.01)
        for _ in range(3):
            assert client.post("/api/analyze/text", json=_TEXT).status_code == 200

    def test_a_stream_gives_its_slot_back(self, app, client):
        app.extensions["omniai_analysis_slots"] = AnalysisSlots(1, wait_seconds=0.5)
        for _ in range(2):
            response = client.post("/api/analyze/text/stream", json=_TEXT)
            assert response.status_code == 200
            response.get_data()


class TestHostileUploads:
    def test_a_zip_bomb_docx_is_refused_before_it_is_parsed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(docx_utils, "MAX_DOCX_UNPACKED_BYTES", 1024 * 1024)
        path = tmp_path / "bomb.docx"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", "a" * (4 * 1024 * 1024))
        with pytest.raises(ValueError, match="too large or too complex"):
            validate_docx_file(str(path))

    def test_an_unreadable_image_error_reveals_no_library_detail(self, tmp_path):
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not really an image")
        with pytest.raises(UploadRejected) as caught:
            validate_image_dimensions(broken)
        message = str(caught.value)
        assert message.startswith("The image could not be read.")
        assert "identify" not in message.lower()


class TestErrorResponses:
    def test_a_wrong_method_gets_a_plain_json_error(self, client):
        response = client.get("/api/analyze/text")
        assert response.status_code == 405
        assert response.get_json() == {"detail": "That method is not allowed here."}

    def test_an_unknown_path_gets_a_plain_json_error(self, client):
        response = client.get("/api/does-not-exist")
        assert response.status_code == 404
        assert response.get_json() == {"detail": "Not found."}


class TestResponseHeaders:
    def test_the_page_is_isolated_from_other_origins(self, client):
        headers = client.get("/").headers
        assert headers["Cross-Origin-Opener-Policy"] == "same-origin"
        assert headers["Cross-Origin-Resource-Policy"] == "same-origin"

    def test_api_responses_are_never_cached(self, client):
        assert client.get("/api/health").headers["Cache-Control"] == "no-store"

    def test_hsts_is_sent_over_https_only(self, client):
        assert "Strict-Transport-Security" not in client.get("/").headers
        assert "Strict-Transport-Security" in client.get("/", base_url="https://localhost").headers

    def test_a_forwarded_https_claim_needs_a_trusted_proxy(self, client):
        headers = client.get("/", headers={"X-Forwarded-Proto": "https"}).headers
        assert "Strict-Transport-Security" not in headers
