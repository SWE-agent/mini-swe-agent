"""Tests for oauth_page.py HTML generation."""

from minisweagent.oauth.oauth_page import oauth_error_html, oauth_success_html


def test_success_html_contains_message():
    out = oauth_success_html("You may close this window.")
    assert "You may close this window." in out
    assert "Authentication successful" in out


def test_error_html_contains_message():
    out = oauth_error_html("Something went wrong.")
    assert "Something went wrong." in out
    assert "Authentication failed" in out


def test_error_html_with_details():
    out = oauth_error_html("Authentication failed.", "Error: bad_state")
    assert "bad_state" in out


def test_success_html_escapes_user_content():
    out = oauth_success_html("<script>alert('xss')</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_error_html_escapes_message():
    out = oauth_error_html("<img onerror=x>")
    assert "<img" not in out


def test_error_html_escapes_details():
    out = oauth_error_html("msg", "<b>injected</b>")
    assert "<b>" not in out
    assert "&lt;b&gt;" in out


def test_success_html_is_valid_html_document():
    out = oauth_success_html("Done!")
    assert out.startswith("<!doctype html>")
    assert "</html>" in out


def test_error_html_no_details_omits_details_block():
    out = oauth_error_html("fail")
    assert 'class="details"' not in out


def test_error_html_with_details_includes_details_block():
    out = oauth_error_html("fail", "some detail")
    assert 'class="details"' in out
    assert "some detail" in out
