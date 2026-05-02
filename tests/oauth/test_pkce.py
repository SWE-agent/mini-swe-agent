import base64
import hashlib

from minisweagent.oauth.pkce import generate_pkce


def test_generate_pkce_produces_valid_s256_pair():
    pair = generate_pkce()

    assert len(pair.verifier) >= 43
    assert all(c.isalnum() or c in "-_" for c in pair.verifier)

    expected = base64.urlsafe_b64encode(hashlib.sha256(pair.verifier.encode()).digest()).rstrip(b"=").decode()
    assert pair.challenge == expected


def test_generate_pkce_is_random():
    a = generate_pkce()
    b = generate_pkce()
    assert a.verifier != b.verifier
    assert a.challenge != b.challenge
