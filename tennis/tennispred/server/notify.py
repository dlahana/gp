"""Tell the account owner a draft is waiting.

Email now (any SMTP account, e.g. Gmail with an app password), SMS via
Twilio later. Each channel turns on when its settings exist; with none set,
the message is only logged (the approval link is still in the server log).

Email:  SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, NOTIFY_EMAIL, SMTP_FROM (optional)
SMS:    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, NOTIFY_PHONE
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

log = logging.getLogger(__name__)


def send_email(subject: str, body: str) -> bool:
    host, to = os.environ.get("SMTP_HOST"), os.environ.get("NOTIFY_EMAIL")
    user, password = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASSWORD")
    if not all((host, to, user, password)):
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, os.environ.get("SMTP_FROM", user), to
    msg.set_content(body)
    with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587")), timeout=30) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
    return True


def send_sms(body: str) -> bool:
    sid, token = os.environ.get("TWILIO_ACCOUNT_SID"), os.environ.get("TWILIO_AUTH_TOKEN")
    from_, to = os.environ.get("TWILIO_FROM"), os.environ.get("NOTIFY_PHONE")
    if not all((sid, token, from_, to)):
        return False
    from twilio.rest import Client

    Client(sid, token).messages.create(to=to, from_=from_, body=body)
    return True


def notify(subject: str, link: str, details: str = "") -> list[str]:
    """Send on every configured channel. Returns the channels that worked."""
    sent = []
    for name, fn, args in (("email", send_email, (subject, f"{subject}\n\nReview and approve: {link}\n\n{details}")),
                           ("sms", send_sms, (f"{subject}: {link}",))):
        try:
            if fn(*args):
                sent.append(name)
        except Exception:
            log.exception("%s notification failed", name)
    if not sent:
        log.warning("no notification channel configured; approve here: %s", link)
    return sent
