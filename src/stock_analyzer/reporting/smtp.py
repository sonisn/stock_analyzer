"""SMTP transport — reads credentials from env, sends HTML or plain mail."""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage

from ..logging import get_logger

logger = get_logger(__name__)

SMTP_TIMEOUT_SECONDS = 15


def _required_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


class SmtpServer:
    """SMTP client that authenticates via environment variables.

    Required: SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD.
    Optional: SMTP_PORT (default 587), SMTP_FROM (default = username),
    SMTP_USE_SSL ("true" for SSL on 465, otherwise STARTTLS on 587).
    """

    def __init__(self) -> None:
        self.host = _required_env("SMTP_HOST")
        self.port = int(os.environ.get("SMTP_PORT", "587"))
        self.username = _required_env("SMTP_USERNAME")
        self.password = _required_env("SMTP_PASSWORD")
        self.sender = os.environ.get("SMTP_FROM", self.username)
        self.use_ssl = os.environ.get("SMTP_USE_SSL", "false").lower() == "true"
        self.skip_verify = (
            os.environ.get("SMTP_INSECURE_SKIP_VERIFY", "false").lower() == "true"
        )

    def _ssl_context(self) -> ssl.SSLContext:
        ctx = ssl.create_default_context()
        if self.skip_verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def send_email(
        self,
        to: str,
        subject: str,
        content: str,
        *,
        content_type: str = "plain",
        inline_images: dict[str, bytes] | None = None,
    ) -> None:
        """Send an email. If `inline_images` is given (keyed by CID → PNG bytes),
        the message is built as multipart/alternative with a related part so the
        HTML body can reference the images via `cid:<key>`.
        """
        msg = EmailMessage()
        msg["From"] = self.sender
        msg["To"] = to
        msg["Subject"] = subject

        if inline_images and content_type == "html":
            msg.set_content(
                "This message contains images. View it in an HTML-capable client."
            )
            msg.add_alternative(content, subtype="html")
            html_part = next(
                p for p in msg.iter_parts()
                if isinstance(p, EmailMessage)
                and p.get_content_type() == "text/html"
            )
            for cid, img_bytes in inline_images.items():
                html_part.add_related(
                    img_bytes,
                    maintype="image",
                    subtype="png",
                    cid=f"<{cid}>",
                )
        else:
            msg.set_content(content, subtype=content_type)

        logger.info(
            "Sending email to %s (subject=%r, inline_images=%d)",
            to,
            subject,
            len(inline_images) if inline_images else 0,
        )
        context = self._ssl_context()
        if self.use_ssl:
            with smtplib.SMTP_SSL(
                self.host, self.port, context=context, timeout=SMTP_TIMEOUT_SECONDS
            ) as server:
                server.login(self.username, self.password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(
                self.host, self.port, timeout=SMTP_TIMEOUT_SECONDS
            ) as server:
                server.starttls(context=context)
                server.login(self.username, self.password)
                server.send_message(msg)
        logger.info("Email sent")
