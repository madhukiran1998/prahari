"""WhatsApp alerts via the Meta Cloud API, with a console fallback.

Two things worth knowing before the demo:

  * Free-form image messages only reach the owner inside a 24-hour window that
    THEY open by messaging your test number first. Have them send "hi" at the
    start of the meeting and every alert is free. Outside that window you need
    an approved template, which is the pilot's job, not the POC's.
  * Sending must never block the pipeline. Everything here is fire-and-forget:
    a failed send is logged and dropped, never retried into a backlog that
    delivers ten stale alerts at once.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("prahari.notify")

GRAPH = "https://graph.facebook.com/v21.0"
ATTEMPTS = 3
TIMEOUT_S = 15.0


class Notifier:
    def __init__(self, cfg: dict[str, Any]):
        self.enabled = bool(cfg.get("enabled"))
        self.token = str(cfg.get("token") or "")
        self.phone_id = str(cfg.get("phone_id") or "")
        self.to = str(cfg.get("to") or "")
        if self.enabled and not (self.token and self.phone_id and self.to):
            log.warning("whatsapp enabled but not configured - falling back to console")
            self.enabled = False

    async def send_alert(self, title: str, body: str, image: Path | None = None) -> bool:
        if not self.enabled:
            log.info("ALERT (console) | %s | %s", title, body)
            return False
        text = f"*{title}*\n{body}"
        for attempt in range(1, ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
                    media_id = None
                    if image and image.exists():
                        media_id = await self._upload(client, image)
                    await self._send(client, text, media_id)
                log.info("whatsapp alert sent: %s", title)
                return True
            except Exception as exc:
                log.warning("whatsapp attempt %d/%d failed: %s", attempt, ATTEMPTS, exc)
                if attempt < ATTEMPTS:
                    await asyncio.sleep(2 ** (attempt - 1))
        log.error("whatsapp alert dropped after %d attempts: %s", ATTEMPTS, title)
        return False

    async def _upload(self, client: httpx.AsyncClient, image: Path) -> str:
        resp = await client.post(
            f"{GRAPH}/{self.phone_id}/media",
            headers={"Authorization": f"Bearer {self.token}"},
            data={"messaging_product": "whatsapp", "type": "image/jpeg"},
            files={"file": (image.name, image.read_bytes(), "image/jpeg")},
        )
        resp.raise_for_status()
        return resp.json()["id"]

    async def _send(
        self, client: httpx.AsyncClient, text: str, media_id: str | None
    ) -> None:
        if media_id:
            payload = {
                "messaging_product": "whatsapp",
                "to": self.to,
                "type": "image",
                "image": {"id": media_id, "caption": text[:1024]},
            }
        else:
            payload = {
                "messaging_product": "whatsapp",
                "to": self.to,
                "type": "text",
                "text": {"body": text[:4096]},
            }
        resp = await client.post(
            f"{GRAPH}/{self.phone_id}/messages",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
