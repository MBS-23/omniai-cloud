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
    def test_local_default_is_offline(self, clean_env):
        assert config._resolve_profile() == "offline"

    def test_serverless_default_is_lite(self, clean_env):
        """A serverless host cannot hold the model weights, so default to lite.

        Without this, a deployment missing OMNIAI_PROFILE would try to import
        PyTorch at startup and every request would fail.
        """
        clean_env.setenv("VERCEL", "1")
        assert config._resolve_profile() == "lite"

    def test_explicit_profile_wins_on_serverless(self, clean_env):
        clean_env.setenv("VERCEL", "1")
        clean_env.setenv("OMNIAI_PROFILE", "full")
        assert config._resolve_profile() == "full"

    @pytest.mark.parametrize("raw,expected", [
        ("offline", "offline"), ("local", "offline"), ("full", "full"),
        ("lite", "lite"), ("LITE", "lite"), ("anything-else", "lite"),
    ])
    def test_profile_aliases(self, clean_env, raw, expected):
        clean_env.setenv("OMNIAI_PROFILE", raw)
        assert config._resolve_profile() == expected


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
