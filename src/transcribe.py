from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .utils import normalize_text
from .gemini_client import get_client
from .utils import env


@dataclass(frozen=True)
class TranscriptSegment:
    media_path: str
    start_s: float
    end_s: float
    text: str


def _cache_path(cache_dir: Path, media_path: Path) -> Path:
    safe = media_path.as_posix().replace("/", "__")
    return cache_dir / f"{safe}.json"


def transcribe_media_dir(
    media_dir: Path,
    cache_dir: Path,
    whisper_model: str = "gemini-3-flash-preview",
) -> list[TranscriptSegment]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = get_client()

    media_exts = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".mp4", ".mkv", ".mov", ".webm"}
    media_files = [p for p in sorted(media_dir.rglob("*")) if p.is_file() and p.suffix.lower() in media_exts]

    file_active_timeout_s = float(env("GEMINI_FILE_ACTIVE_TIMEOUT_S", "120") or "120")

    all_segments: list[TranscriptSegment] = []
    for media_path in tqdm(media_files, desc="Transcribing media", unit="file"):
        cp = _cache_path(cache_dir, media_path)
        if cp.exists():
            data = json.loads(cp.read_text(encoding="utf-8"))
            for s in data.get("segments", []):
                txt = normalize_text(s.get("text", ""))
                if txt:
                    all_segments.append(
                        TranscriptSegment(
                            media_path=str(media_path),
                            start_s=float(s.get("start", 0.0)),
                            end_s=float(s.get("end", 0.0)),
                            text=txt,
                        )
                    )
            continue

        # Use the Gemini Files API for large media.
        uploaded = client.files.upload(file=str(media_path))
        try:
            # Uploaded files may take a bit to become ACTIVE.
            deadline = time.time() + file_active_timeout_s
            while True:
                info = client.files.get(name=uploaded.name)
                state = getattr(info, "state", None)
                state_name = getattr(state, "name", None) or str(state) if state is not None else "UNKNOWN"
                if str(state_name).upper() == "ACTIVE":
                    uploaded = info
                    break
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"Gemini file did not become ACTIVE within {file_active_timeout_s:.0f}s: {uploaded.name} (state={state_name})"
                    )
                time.sleep(1.0)

            prompt = (
                "Transcribe the provided audio/video into clean text. "
                "Do not add commentary. Preserve terminology and abbreviations."
            )
            resp = client.models.generate_content(model=whisper_model, contents=[prompt, uploaded])
            txt = normalize_text(resp.text or "")
        finally:
            # Best-effort cleanup; ignore failures.
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

        seg_out: list[dict] = []
        if txt:
            all_segments.append(TranscriptSegment(media_path=str(media_path), start_s=0.0, end_s=0.0, text=txt))
            seg_out.append({"start": 0.0, "end": 0.0, "text": txt})

        cp.write_text(
            json.dumps(
                {"media_path": str(media_path), "model": whisper_model, "segments": seg_out},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return all_segments

