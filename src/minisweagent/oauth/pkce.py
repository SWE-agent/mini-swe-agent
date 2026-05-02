"""PKCE helpers (RFC 7636).

Ported from pi-mono (packages/ai/src/utils/oauth/pkce.ts).
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass


@dataclass
class PKCEPair:
    verifier: str
    challenge: str


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce() -> PKCEPair:
    verifier_bytes = secrets.token_bytes(32)
    verifier = _b64url(verifier_bytes)
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return PKCEPair(verifier=verifier, challenge=challenge)
