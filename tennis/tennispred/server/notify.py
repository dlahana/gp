"""Text the account owner when a draft needs approval (Twilio SMS)."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def send_sms(body: str) -> bool:
    """Returns True if a text was sent. Without Twilio settings it only logs."""
    sid, token = os.environ.get("TWILIO_ACCOUNT_SID"), os.environ.get("TWILIO_AUTH_TOKEN")
    from_, to = os.environ.get("TWILIO_FROM"), os.environ.get("NOTIFY_PHONE")
    if not all((sid, token, from_, to)):
        log.warning("Twilio not configured; would have texted: %s", body)
        return False
    from twilio.rest import Client

    Client(sid, token).messages.create(to=to, from_=from_, body=body)
    return True
