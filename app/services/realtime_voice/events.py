# -*- coding: utf-8 -*-
"""Realtime voice WebSocket event envelope helpers."""

from __future__ import annotations

import time
from typing import Any


REALTIME_VOICE_SCHEMA_VERSION = "realtime_voice.v1"
REALTIME_VOICE_SUCCESS_STATUS = 20000000


class RealtimeVoiceEventBuilder:
    """Build sequenced realtime-voice events with a stable envelope."""

    def __init__(self, task_id: str, session_id: str | None = None):
        self.task_id = task_id
        self.session_id = session_id or f"session_{task_id}"
        self._seq = 0

    def build(
        self,
        event: str,
        *,
        payload: dict[str, Any] | None = None,
        status: int = REALTIME_VOICE_SUCCESS_STATUS,
        **legacy_fields: Any,
    ) -> dict[str, Any]:
        """Return an event preserving legacy top-level fields plus v1 payload."""
        self._seq += 1
        message = {
            "event": event,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "seq": self._seq,
            "server_ts_ms": int(time.time() * 1000),
            "schema_version": REALTIME_VOICE_SCHEMA_VERSION,
            "status": status,
            "payload": payload or {},
        }
        message.update(legacy_fields)
        return message

