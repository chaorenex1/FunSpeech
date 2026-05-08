# -*- coding: utf-8 -*-
"""Pressure-test WS /ws/v1/realtime/voice with a local WAV file."""

from __future__ import annotations

import argparse
import asyncio
import audioop
import json
import statistics
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import websockets


@dataclass
class SessionMetrics:
    request_id: str
    concurrency: int
    success: bool = False
    error: str = ""
    started_at: float = 0.0
    configured_at: float | None = None
    first_asr_at: float | None = None
    first_commit_at: float | None = None
    first_tts_job_at: float | None = None
    first_audio_meta_at: float | None = None
    first_binary_at: float | None = None
    completed_at: float | None = None
    events: dict[str, int] = field(default_factory=dict)
    backpressure_reasons: dict[str, int] = field(default_factory=dict)
    binary_bytes: int = 0
    binary_chunks: int = 0
    backpressure_events: int = 0
    error_events: int = 0

    def mark_event(self, event: str) -> None:
        self.events[event] = self.events.get(event, 0) + 1

    def mark_backpressure(self, reason: str) -> None:
        self.backpressure_reasons[reason] = self.backpressure_reasons.get(reason, 0) + 1

    def latency_ms(self, value: float | None) -> float | None:
        if value is None:
            return None
        return (value - self.started_at) * 1000

    def to_summary(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "configured_latency_ms": self.latency_ms(self.configured_at),
            "first_asr_latency_ms": self.latency_ms(self.first_asr_at),
            "first_commit_latency_ms": self.latency_ms(self.first_commit_at),
            "first_tts_job_latency_ms": self.latency_ms(self.first_tts_job_at),
            "first_audio_meta_latency_ms": self.latency_ms(self.first_audio_meta_at),
            "first_binary_latency_ms": self.latency_ms(self.first_binary_at),
            "total_time_ms": self.latency_ms(self.completed_at),
        }


def load_wav_as_pcm16_mono_16k(path: Path, max_seconds: float | None) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        if max_seconds:
            frames = min(frames, int(sample_rate * max_seconds))
        raw = wav.readframes(frames)

    if sample_width != 2:
        raw = audioop.lin2lin(raw, sample_width, 2)
        sample_width = 2
    if channels == 2:
        raw = audioop.tomono(raw, sample_width, 0.5, 0.5)
    elif channels != 1:
        raise ValueError(f"unsupported channel count: {channels}")
    if sample_rate != 16000:
        raw, _ = audioop.ratecv(raw, sample_width, 1, sample_rate, 16000, None)
        sample_rate = 16000

    duration_s = len(raw) / 2 / sample_rate
    return raw, duration_s


async def receive_loop(ws, metrics: SessionMetrics, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        now = time.perf_counter()
        if isinstance(message, bytes):
            metrics.binary_chunks += 1
            metrics.binary_bytes += len(message)
            if metrics.first_binary_at is None:
                metrics.first_binary_at = now
            continue

        data = json.loads(message)
        event = data.get("event", "")
        metrics.mark_event(event)

        if event == "configured" and metrics.configured_at is None:
            metrics.configured_at = now
        elif event in {"asr.hypothesis", "asr_result"} and metrics.first_asr_at is None:
            metrics.first_asr_at = now
        elif event == "asr.text_committed" and metrics.first_commit_at is None:
            metrics.first_commit_at = now
        elif event == "tts.job_started" and metrics.first_tts_job_at is None:
            metrics.first_tts_job_at = now
        elif event == "tts.first_audio" and metrics.first_audio_meta_at is None:
            metrics.first_audio_meta_at = now
        elif event == "session_completed":
            metrics.completed_at = now
            metrics.success = metrics.error_events == 0
            stop_event.set()
        elif event in {"session.error", "error"}:
            metrics.error_events += 1
            payload = data.get("payload") or {}
            metrics.error = data.get("message") or payload.get("message") or json.dumps(data, ensure_ascii=False)
            stop_event.set()

        if event.startswith("backpressure") or event == "backpressure":
            metrics.backpressure_events += 1
            payload = data.get("payload") or {}
            metrics.mark_backpressure(
                str(payload.get("reason") or data.get("type") or "unknown")
            )


async def run_one(
    index: int,
    concurrency: int,
    ws_url: str,
    audio: bytes,
    voice: str,
    chunk_ms: int,
    realtime_factor: float,
    drain_seconds: float,
    timeout: float,
) -> SessionMetrics:
    metrics = SessionMetrics(
        request_id=f"rv-{concurrency}-{index}",
        concurrency=concurrency,
        started_at=time.perf_counter(),
    )
    stop_event = asyncio.Event()
    try:
        async with websockets.connect(ws_url, ping_interval=None, max_size=32 * 1024 * 1024) as ws:
            receiver = asyncio.create_task(receive_loop(ws, metrics, stop_event))
            configure = {
                "event": "configure",
                "voice_name": voice,
                "format": "pcm",
                "sample_rate": 16000,
                "pipeline": "asr_tts",
                "parameters": {"volume": 50, "speech_rate": 0, "pitch_rate": 0},
            }
            await ws.send(json.dumps(configure, ensure_ascii=False))

            chunk_bytes = int(16000 * chunk_ms / 1000) * 2
            sleep_s = chunk_ms / 1000 * realtime_factor
            for offset in range(0, len(audio), chunk_bytes):
                if stop_event.is_set():
                    break
                await ws.send(audio[offset : offset + chunk_bytes])
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

            deadline = time.perf_counter() + drain_seconds
            while not stop_event.is_set() and time.perf_counter() < deadline:
                await asyncio.sleep(0.1)
            if not stop_event.is_set():
                await ws.send(json.dumps({"event": "stop"}))
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=5)
                except asyncio.TimeoutError:
                    pass
            receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass

        if metrics.completed_at is None:
            metrics.completed_at = time.perf_counter()
        if not metrics.error:
            metrics.success = True
    except Exception as exc:
        metrics.error = f"{type(exc).__name__}: {exc}"
        metrics.completed_at = time.perf_counter()
    return metrics


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, int(round((len(values) - 1) * pct)))
    return values[index]


def aggregate(level: int, metrics: list[SessionMetrics]) -> dict[str, Any]:
    summaries = [m.to_summary() for m in metrics]
    first_binary = [s["first_binary_latency_ms"] for s in summaries if s["first_binary_latency_ms"] is not None]
    first_asr = [s["first_asr_latency_ms"] for s in summaries if s["first_asr_latency_ms"] is not None]
    total = [s["total_time_ms"] for s in summaries if s["total_time_ms"] is not None]
    return {
        "concurrency": level,
        "total": len(metrics),
        "success": sum(1 for m in metrics if m.success),
        "failed": sum(1 for m in metrics if not m.success),
        "backpressure_events": sum(m.backpressure_events for m in metrics),
        "error_events": sum(m.error_events for m in metrics),
        "first_asr_ms": stats(first_asr),
        "first_binary_ms": stats(first_binary),
        "total_time_ms": stats(total),
        "binary_bytes_total": sum(m.binary_bytes for m in metrics),
        "event_counts": merge_event_counts(metrics),
        "backpressure_reasons": merge_backpressure_reasons(metrics),
        "sessions": summaries,
    }


def stats(values: list[float]) -> dict[str, float | None]:
    return {
        "avg": statistics.mean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def merge_event_counts(metrics: list[SessionMetrics]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for item in metrics:
        for event, count in item.events.items():
            merged[event] = merged.get(event, 0) + count
    return dict(sorted(merged.items()))


def merge_backpressure_reasons(metrics: list[SessionMetrics]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for item in metrics:
        for reason, count in item.backpressure_reasons.items():
            merged[reason] = merged.get(reason, 0) + count
    return dict(sorted(merged.items()))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://10.0.0.96:8000")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--voice", default="中文女")
    parser.add_argument("--levels", default="1,2,4")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--realtime-factor", type=float, default=0.5)
    parser.add_argument("--drain-seconds", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    audio, duration_s = load_wav_as_pcm16_mono_16k(Path(args.audio), args.duration)
    ws_url = args.base_url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + "/ws/v1/realtime/voice"
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]

    report = {
        "ws_url": ws_url,
        "audio": str(Path(args.audio).resolve()),
        "voice": args.voice,
        "duration_s": duration_s,
        "chunk_ms": args.chunk_ms,
        "realtime_factor": args.realtime_factor,
        "levels": [],
    }

    for level in levels:
        started = time.perf_counter()
        tasks = [
            run_one(i, level, ws_url, audio, args.voice, args.chunk_ms, args.realtime_factor, args.drain_seconds, args.timeout)
            for i in range(level)
        ]
        metrics = await asyncio.gather(*tasks)
        item = aggregate(level, list(metrics))
        item["wall_time_s"] = time.perf_counter() - started
        report["levels"].append(item)
        print(json.dumps({k: v for k, v in item.items() if k != "sessions"}, ensure_ascii=False, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report_saved={args.output}")


if __name__ == "__main__":
    asyncio.run(main())
