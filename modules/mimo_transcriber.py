"""MiMo text recognition with genuine local Whisper timing for captions.

API: https://mimo.mi.com/docs/en-US/api/audio/Speech-Recognition
No MiMo word timestamps are invented or assigned to Whisper words.
"""
import base64
import copy
import math
import os
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

MIMO_MODEL = "mimo-v2.5-asr"
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MAX_ENCODED_BYTES = 10_000_000


class MimoError(RuntimeError):
    pass


def validate_transcription_engine(value):
    if value not in ("whisper", "mimo"):
        raise ValueError("Unknown transcription engine; choose Whisper Local or Xiaomi MiMo API.")
    return value


class MimoTranscriber:
    def transcribe_segments(self, video_path, timed_transcript, language=None):
        key = os.environ.get("MIMO_API_KEY", "").strip()
        if not key:
            raise MimoError("MIMO_API_KEY is not configured.")
        if language and language not in ("auto", "en", "zh"):
            raise MimoError("MiMo supports language hints auto, en and zh only.")
        result = copy.deepcopy(timed_transcript)
        try:
            # Bounded requests; never propagate SDK exceptions containing request details.
            with OpenAI(api_key=key, base_url=MIMO_BASE_URL,
                        timeout=120.0, max_retries=0) as client:
                with tempfile.TemporaryDirectory(prefix="clipper-mimo-") as directory:
                    audio_path = Path(directory) / "segment.wav"
                    for segment in result["segments"]:
                        start, end = float(segment["start"]), float(segment["end"])
                        if not all(map(math.isfinite, (start, end))) or start < 0 or end <= start:
                            raise MimoError("Local timing contains an invalid segment.")
                        # 16 kHz mono PCM + Base64 stays below 10 MB for 180 seconds.
                        if end - start > 180:
                            raise MimoError("A local speech segment exceeds the safe MiMo audio limit.")
                        subprocess.run([
                            "ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", str(start),
                            "-i", str(video_path), "-t", str(end - start), "-vn",
                            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio_path),
                        ], check=True, capture_output=True, timeout=120)
                        encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
                        if len(encoded) > MAX_ENCODED_BYTES:
                            raise MimoError("MiMo audio exceeds the 10 MB Base64 limit.")
                        response = client.chat.completions.create(
                            model=MIMO_MODEL,
                            messages=[{"role": "user", "content": [{"type": "input_audio",
                                "input_audio": {"data": "data:audio/wav;base64," + encoded}}]}],
                            extra_body={"asr_options": {"language": language or "auto"}},
                        )
                        choice = response.choices[0]
                        text = choice.message.content
                        if choice.finish_reason != "stop" or not isinstance(text, str) or not text.strip():
                            raise MimoError("MiMo returned an empty or incomplete transcript.")
                        segment["whisper_text"] = segment["text"]
                        segment["text"] = text.strip()
            result["full_text"] = " ".join(s["text"] for s in result["segments"])
            result["transcription_engine"] = "mimo"
            result["text_provider"] = MIMO_MODEL
            result["timing_provider"] = "whisper"
            result["subtitle_provider"] = "whisper"
            return result
        except MimoError:
            raise
        except Exception:
            raise MimoError("MiMo request or audio preparation failed; check connectivity, API access and FFmpeg.") from None


def transcribe_with_engine(local_transcriber, video_path, engine="whisper", *,
                           force=False, language=None, notice=None):
    validate_transcription_engine(engine)
    notify = notice or (lambda message: None)
    if engine == "mimo":
        notify("MiMo uses local Whisper timing. Clip analysis uses MiMo text; subtitle words and timing remain Whisper.")
    # Retain the existing Whisper cache; MiMo never overwrites it.
    try:
        local = local_transcriber.transcribe(video_path, force=force, language=language)
    except Exception:
        if engine == "mimo":
            raise RuntimeError(
                "MiMo mode requires local Whisper timing, but Whisper failed. "
                "Check the local Whisper model and audio dependencies; no audio was sent to MiMo."
            ) from None
        raise
    if engine == "whisper":
        return local
    try:
        if not local.get("segments"):
            raise MimoError("Whisper found no timed speech segments.")
        return MimoTranscriber().transcribe_segments(video_path, local, language)
    except MimoError as exc:
        notify(f"{exc} Falling back to Whisper Local for transcript and subtitles.")
        return local
