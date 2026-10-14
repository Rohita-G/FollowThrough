#!/usr/bin/env python3
"""Parse raw MIME/RFC-822 email files into structured dictionaries."""

import argparse
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class HTMLToTextParser(HTMLParser):
    BLOCK_TAGS = {"address", "article", "br", "div", "footer", "h1", "h2", "h3", "li", "p", "section", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def decode_part(part: Any) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raw = part.get_payload()
        return raw if isinstance(raw, str) else ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def extract_body(message: Any) -> Tuple[str, str]:
    """Extract plain text or html fallback body from email object."""
    if message.is_multipart():
        plain_parts = [
            part for part in message.walk()
            if part.get_content_type() == "text/plain" and part.get_content_disposition() != "attachment"
        ]
        if plain_parts:
            return "\n\n".join(decode_part(p) for p in plain_parts).replace("\r\n", "\n").replace("\r", "\n"), "plain"
        html_parts = [
            part for part in message.walk()
            if part.get_content_type() == "text/html" and part.get_content_disposition() != "attachment"
        ]
        if html_parts:
            parser = HTMLToTextParser()
            for part in html_parts:
                try:
                    parser.feed(decode_part(part))
                except Exception:
                    pass
            return "".join(parser.parts).replace("\r\n", "\n").replace("\r", "\n"), "html_fallback"
        return "", "no_text_part"
    if message.get_content_type() == "text/html":
        parser = HTMLToTextParser()
        parser.feed(decode_part(message))
        return "".join(parser.parts).replace("\r\n", "\n").replace("\r", "\n"), "html_fallback"
    return decode_part(message).replace("\r\n", "\n").replace("\r", "\n"), "plain"


def parse_email(file_path: Path, maildir_base: Optional[Path] = None) -> Dict[str, Any]:
    """Parse a single raw email file and return structured header & body metadata."""
    file_path = Path(file_path)
    rel_path = str(file_path.relative_to(maildir_base)) if maildir_base else str(file_path)
    mailbox = rel_path.split("/", 1)[0] if "/" in rel_path else "unknown"

    raw_bytes = file_path.read_bytes()
    message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    body, body_format = extract_body(message)

    header_names = ("Message-ID", "In-Reply-To", "References", "From", "To", "Cc", "Subject", "Date")
    return {
        "message_id": str(message.get("Message-ID", "") or "").strip(),
        "in_reply_to": str(message.get("In-Reply-To", "") or "").strip(),
        "references": str(message.get("References", "") or "").strip(),
        "from": str(message.get("From", "") or "").strip(),
        "to": str(message.get("To", "") or "").strip(),
        "cc": str(message.get("Cc", "") or "").strip(),
        "subject": str(message.get("Subject", "") or "").strip(),
        "date": str(message.get("Date", "") or "").strip(),
        "body": body,
        "body_format": body_format,
        "source_relative_path": rel_path,
        "mailbox": mailbox,
        "header_presence": {name: message.get(name) is not None for name in header_names},
    }
