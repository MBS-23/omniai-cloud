"""Configuration resolution, including the serverless defaults."""

from __future__ import annotations

import pytest

from backend import config


@pytest.fixture
def clean_env(monkeypatch):
    for name in (
        "OMNIAI_PROFILE", "PROFILE", "VERCEL", "AWS_LAMBDA_FUNCTION_NAME",
        "OMNIAI_HOSTED_PROVIDER", "OMNIAI_GEMINI_API_KEY", "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class TestProfileResolution:
    @staticmethod
    def _installed_only(monkeypatch, *present: str) -> None:
        monkeypatch.setattr(config, "_installed", lambda *modules: all(m in present for m in modules))

    def test_default_is_the_paper_profile_when_its_packages_are_installed(self, clean_env):
        self._installed_only(clean_env, *config.PAPER_MODULES, *config.OFFLINE_MODULES)
        assert config._resolve_profile() == "paper"

    def test_default_falls_back_to_offline_without_paddleocr(self, clean_env):
        """PaddlePaddle has no Python 3.14 wheels; offline still runs there."""
        self._installed_only(clean_env, *config.OFFLINE_MODULES)
        assert config._resolve_profile() == "offline"

    def test_default_is_lite_with_no_local_models_installed(self, clean_env):
        self._installed_only(clean_env)
        assert config._resolve_profile() == "lite"

    def test_serverless_default_is_lite(self, clean_env):
        """A serverless host cannot hold the model weights, so default to lite.

        Without this, a deployment missing OMNIAI_PROFILE would try to import
        PyTorch at startup and every request would fail.
        """
        clean_env.setenv("VERCEL", "1")
        assert config._resolve_profile() == "lite"

    def test_explicit_profile_wins_on_serverless(self, clean_env):
        clean_env.setenv("VERCEL", "1")
        clean_env.setenv("OMNIAI_PROFILE", "paper")
        assert config._resolve_profile() == "paper"

    @pytest.mark.parametrize("raw,expected", [
        ("paper", "paper"), ("full", "paper"), ("PAPER", "paper"),
        ("offline", "offline"), ("local", "offline"),
        ("lite", "lite"), ("LITE", "lite"),
    ])
    def test_profile_aliases(self, clean_env, raw, expected):
        clean_env.setenv("OMNIAI_PROFILE", raw)
        assert config._resolve_profile() == expected

    def test_unknown_profile_falls_back_to_the_automatic_choice(self, clean_env):
        self._installed_only(clean_env)
        clean_env.setenv("OMNIAI_PROFILE", "anything-else")
        assert config._resolve_profile() == "lite"


class TestLocalModelSettings:
    def test_ocr_languages_come_from_a_comma_list(self, clean_env):
        clean_env.setenv("OMNIAI_OCR_LANGS", " en, TE ,hi ")
        assert config.reload_settings().ocr_langs == ("en", "te", "hi")

    def test_ocr_defaults_to_english(self, clean_env):
        clean_env.delenv("OMNIAI_OCR_LANGS", raising=False)
        assert config.reload_settings().ocr_langs == ("en",)

    @pytest.mark.parametrize("raw,expected", [("all", "all"), ("none", "none"), ("bogus", "core"), ("", "core")])
    def test_preload_values(self, clean_env, raw, expected):
        clean_env.setenv("OMNIAI_PRELOAD", raw)
        assert config.reload_settings().preload == expected

    def test_summariser_is_local_t5_unless_mistral_is_asked_for(self, clean_env):
        clean_env.delenv("OMNIAI_SUMMARIZER", raising=False)
        assert config.reload_settings().summarizer == "t5"
        clean_env.setenv("OMNIAI_SUMMARIZER", "Mistral")
        assert config.reload_settings().summarizer == "mistral"


class TestHostedProvider:
    def test_prefers_gemini_when_its_key_is_present(self, clean_env):
        clean_env.setenv("OMNIAI_GEMINI_API_KEY", "g-key")
        assert config._resolve_hosted_provider() == "gemini"

    def test_falls_back_to_mistral_without_a_gemini_key(self, clean_env):
        assert config._resolve_hosted_provider() == "mistral"

    def test_explicit_provider_wins(self, clean_env):
        clean_env.setenv("OMNIAI_GEMINI_API_KEY", "g-key")
        clean_env.setenv("OMNIAI_HOSTED_PROVIDER", "mistral")
        assert config._resolve_hosted_provider() == "mistral"

    def test_legacy_gemini_variable_is_honoured(self, clean_env):
        clean_env.setenv("GEMINI_API_KEY", "g-key")
        assert config._resolve_hosted_provider() == "gemini"


class TestServerlessPaths:
    def test_writable_dirs_move_to_tmp_on_serverless(self, clean_env):
        """Vercel's code directory is read-only; only /tmp is writable."""
        clean_env.setenv("VERCEL", "1")
        settings = config.reload_settings()
        assert "omniai_uploads" in str(settings.upload_dir)
        assert str(settings.upload_dir).startswith(str(settings.upload_dir.anchor))
        assert settings.project_root not in settings.upload_dir.parents

    def test_health_description_never_includes_keys(self, clean_env):
        clean_env.setenv("OMNIAI_GEMINI_API_KEY", "super-secret-gemini-key")
        described = str(config.reload_settings().describe())
        assert "super-secret-gemini-key" not in described


class TestKeyLookup:
    """Keys are found under common alternative names, and blanks never hide them."""

    def test_google_api_key_is_accepted_for_gemini(self, clean_env):
        """GOOGLE_API_KEY is what Google's own SDKs read."""
        clean_env.setenv("GOOGLE_API_KEY", "g-key")
        assert config._raw("OMNIAI_GEMINI_API_KEY") == "g-key"
        assert config._resolve_hosted_provider() == "gemini"

    def test_blank_primary_does_not_hide_a_set_alias(self, clean_env):
        clean_env.setenv("OMNIAI_GEMINI_API_KEY", "   ")
        clean_env.setenv("GEMINI_API_KEY", "real-key")
        assert config._raw("OMNIAI_GEMINI_API_KEY") == "real-key"

    def test_surrounding_quotes_and_spaces_are_stripped(self, clean_env):
        """Dashboards and .env files both tempt people into quoting values."""
        clean_env.setenv("OMNIAI_GEMINI_API_KEY", '  "quoted-key"  ')
        assert config._raw("OMNIAI_GEMINI_API_KEY") == "quoted-key"


class TestKeyDiagnostics:
    def test_reports_a_blank_variable_by_name(self, clean_env):
        clean_env.setenv("OMNIAI_GEMINI_API_KEY", "")
        assert "OMNIAI_GEMINI_API_KEY" in config.key_diagnostics()["set_but_empty"]

    def test_reports_a_misnamed_key_by_name(self, clean_env):
        clean_env.setenv("GEMINI_KEY", "a-value")
        assert "GEMINI_KEY" in config.key_diagnostics()["similar_names_found"]

    def test_never_includes_a_value(self, clean_env):
        clean_env.setenv("GEMINI_KEY", "do-not-leak-this-value")
        clean_env.setenv("OMNIAI_GEMINI_API_KEY", "nor-this-one")
        assert "do-not-leak" not in str(config.key_diagnostics())
        assert "nor-this-one" not in str(config.key_diagnostics())


def test_health_shows_diagnostics_only_while_unconfigured(client, backend):
    """A configured deployment must not advertise its variable names."""
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok"
    assert "diagnostics" not in body
