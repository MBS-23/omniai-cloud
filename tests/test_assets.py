"""Checks on the stylesheet and template that catch whole-page breakage.

These are cheap static checks, not a substitute for looking at the page. They
exist because the failures they catch are silent: the page still renders, still
returns 200, and every Python test still passes, while a control the user needs
is invisible.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "backend" / "static"
TEMPLATES = Path(__file__).resolve().parents[1] / "backend" / "templates"

CSS = (STATIC / "css" / "omniai.css").read_text(encoding="utf-8")
HTML = (TEMPLATES / "index.html").read_text(encoding="utf-8")
APP_JS = (STATIC / "js" / "omniai.js").read_text(encoding="utf-8")


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


class TestFormFieldsAreNeverHidden:
    """`.field` wraps every labelled control: label, input, hint.

    A previous version also used `.field` for the decorative WebGL canvas, and
    two rules hid that canvas -- on screens under 640px and under
    prefers-reduced-motion. Both rules therefore hid every text input, select
    and switch on the page. The form was invisible on every phone.
    """

    def test_no_rule_sets_display_none_on_the_field_class(self):
        css = _strip_comments(CSS)
        offenders = []

        # Find each selector block, then check any that targets bare `.field`.
        for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
            selector, body = match.group(1).strip(), match.group(2)
            targets_field = re.search(r"(^|[\s,>+~])\.field(?![\w-])", selector)
            if targets_field and re.search(r"display\s*:\s*none", body):
                offenders.append(selector.replace("\n", " ").strip()[:70])

        assert not offenders, (
            "These rules hide `.field`, which would hide every form control: "
            f"{offenders}"
        )

    def test_the_canvas_does_not_use_the_field_class(self):
        canvas = re.search(r"<canvas[^>]*>", HTML)
        assert canvas, "the 3D canvas element is missing from the template"
        assert 'class="field"' not in canvas.group(0), (
            "the canvas must not reuse the `.field` form class"
        )
        assert "backdrop__field" in canvas.group(0)


class TestTemplateContract:
    """Every element the client script reaches for must exist in the markup."""

    def test_every_getelementbyid_target_exists(self):
        referenced = set(re.findall(r"\$\('([A-Za-z0-9_-]+)'\)", APP_JS))
        referenced |= set(re.findall(r"getElementById\('([A-Za-z0-9_-]+)'\)", APP_JS))
        present = set(re.findall(r'id="([A-Za-z0-9_-]+)"', HTML))

        missing = sorted(referenced - present)
        assert not missing, f"omniai.js looks for ids the template does not define: {missing}"

    @pytest.mark.parametrize(
        "element_id",
        ["textInput", "sourceLanguage", "targetLanguage", "fileInput",
         "analyzeBtn", "themeToggle", "panels", "figures", "explainBody"],
    )
    def test_essential_controls_are_present(self, element_id):
        assert f'id="{element_id}"' in HTML


class TestAccessibility:
    def test_reduced_motion_is_respected(self):
        assert "prefers-reduced-motion" in CSS

    def test_focus_is_never_removed_without_replacement(self):
        css = _strip_comments(CSS)
        # `outline: none` is acceptable only where a box-shadow ring replaces it.
        for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
            body = match.group(2)
            if re.search(r"outline\s*:\s*(none|0)", body):
                assert "box-shadow" in body, (
                    f"`{match.group(1).strip()[:50]}` removes the focus outline "
                    "without providing a replacement ring"
                )

    def test_every_theme_defines_the_same_token_set(self):
        """A token missing from one theme silently falls back to the other."""
        def tokens(block: str) -> set[str]:
            return set(re.findall(r"(--[a-z0-9-]+)\s*:", block))

        dark = re.search(r":root\s*\{(.*?)\}", CSS, re.S)
        light = re.search(r':root\[data-theme="light"\]\s*\{(.*?)\}', CSS, re.S)
        assert dark and light, "both themes must be defined"

        # Light overrides colours only; the type scale, spacing and motion are
        # shared. `--text-2` is a colour, `--text-lg` is a font size, so the
        # pattern has to distinguish them rather than match on the prefix.
        colour_like = re.compile(
            r"--(bg|surface|border|accent|ok|warn|danger|info|grid|sheen|shadow)"
            r"|--text(-[0-9]+)?$"
        )
        dark_colours = {t for t in tokens(dark.group(1)) if colour_like.match(t)}
        light_colours = {t for t in tokens(light.group(1)) if colour_like.match(t)}

        missing = sorted(dark_colours - light_colours)
        assert not missing, f"light theme does not redefine: {missing}"


class TestNoSecretsInAssets:
    """Client-side assets must never carry a credential."""

    @pytest.mark.parametrize("name", ["omniai.css", "index.html", "omniai.js"])
    def test_no_api_key_shaped_strings(self, name):
        content = {"omniai.css": CSS, "index.html": HTML, "omniai.js": APP_JS}[name]
        patterns = [
            r"sk-[A-Za-z0-9]{16,}",
            r"Bearer\s+[A-Za-z0-9._-]{16,}",
            r"api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9._-]{8,}",
        ]
        for pattern in patterns:
            found = re.search(pattern, content, re.I)
            assert not found, f"{name} contains something key-shaped: {found.group(0)[:24]}..."


class TestContentSecurityPolicyCompatibility:
    """The CSP allows scripts only from this origin, so none may be inline.

    An inline theme script once shipped here; the browser blocked it under
    `script-src 'self'`, and the dark-flash it existed to prevent happened
    anyway. The Python tests could not see it and the page still returned 200.
    Only a real browser's console reported it.
    """

    def test_no_inline_executable_scripts(self):
        offenders = []
        for match in re.finditer(r"<script\b([^>]*)>(.*?)</script>", HTML, re.S):
            attributes, body = match.group(1), match.group(2).strip()
            if "src=" in attributes:
                continue
            # Data blocks are not executed, so the CSP does not apply to them.
            if re.search(r'type\s*=\s*"application/(ld\+)?json"', attributes):
                continue
            if body:
                offenders.append(body[:60].replace("\n", " "))
        assert not offenders, f"inline scripts would be blocked by the CSP: {offenders}"

    def test_no_inline_event_handlers(self):
        handlers = re.findall(r"\son[a-z]+\s*=\s*\"", HTML)
        assert not handlers, f"inline event handlers are blocked by the CSP: {handlers}"
