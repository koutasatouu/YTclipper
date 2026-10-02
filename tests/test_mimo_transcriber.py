import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules import mimo_transcriber as mimo


@pytest.fixture
def timed():
    return {"language": "en", "full_text": "Hello", "segments": [
        {"start": 1.25, "end": 2.75, "text": "Hello", "words": [
            {"word": "Hello", "start": 1.3, "end": 2.6}]}]}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "test-placeholder")
    factory = MagicMock()
    client = factory.return_value.__enter__.return_value
    client.chat.completions.create.return_value = SimpleNamespace(choices=[
        SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="Hello there!"))])
    monkeypatch.setattr(mimo, "OpenAI", factory)
    def extract(command, **kwargs):
        Path(command[-1]).write_bytes(b"mock-wav")
    ffmpeg = MagicMock(side_effect=extract)
    monkeypatch.setattr(mimo.subprocess, "run", ffmpeg)
    return factory, client, ffmpeg


def test_official_request_and_real_timing_preserved(api, timed):
    factory, client, ffmpeg = api
    original = copy.deepcopy(timed)
    result = mimo.MimoTranscriber().transcribe_segments("video.mp4", timed)
    request = client.chat.completions.create.call_args.kwargs
    assert factory.call_args.kwargs["base_url"] == "https://api.xiaomimimo.com/v1"
    assert request["model"] == "mimo-v2.5-asr"
    assert request["extra_body"] == {"asr_options": {"language": "auto"}}
    assert request["messages"][0]["content"][0]["input_audio"]["data"].startswith("data:audio/wav;base64,")
    assert result["full_text"] == "Hello there!"
    assert result["segments"][0]["words"] == timed["segments"][0]["words"]
    assert result["segments"][0]["start"] == 1.25
    assert timed == original
    assert result["subtitle_provider"] == "whisper"
    command = ffmpeg.call_args.args[0]
    assert command[command.index("-ss") + 1] == "1.25"
    assert command[command.index("-t") + 1] == "1.5"
    assert not Path(command[-1]).exists()  # Temporary audio cleaned up.


def test_default_never_calls_api(api, timed):
    local = MagicMock()
    local.transcribe.return_value = timed
    assert mimo.transcribe_with_engine(local, "video") is timed
    api[0].assert_not_called()


def test_missing_key_falls_back_without_network(monkeypatch, api, timed):
    monkeypatch.delenv("MIMO_API_KEY")
    local = MagicMock()
    local.transcribe.return_value = timed
    notices = []
    assert mimo.transcribe_with_engine(local, "video", "mimo", notice=notices.append) is timed
    assert "MIMO_API_KEY" in notices[-1]
    api[0].assert_not_called()
    api[2].assert_not_called()


@pytest.mark.parametrize("failure", ["exception", "empty", "truncated", "oversize", "duration"])
def test_failures_are_atomic_and_sanitized(monkeypatch, api, timed, failure):
    client = api[1]
    if failure == "exception":
        client.chat.completions.create.side_effect = RuntimeError("test-placeholder secret request body")
    elif failure in ("empty", "truncated"):
        choice = client.chat.completions.create.return_value.choices[0]
        choice.message.content = "" if failure == "empty" else "partial"
        choice.finish_reason = "stop" if failure == "empty" else "length"
    elif failure == "oversize":
        monkeypatch.setattr(mimo, "MAX_ENCODED_BYTES", 1)
    else:
        timed["segments"][0]["end"] = 999
    local = MagicMock()
    local.transcribe.return_value = timed
    notices = []
    assert mimo.transcribe_with_engine(local, "video", "mimo", notice=notices.append) is timed
    assert "Falling back" in notices[-1]
    assert "test-placeholder" not in str(notices)
    assert "secret request body" not in str(notices)
    assert timed["full_text"] == "Hello"


def test_process_routes_selected_engine_to_analysis(monkeypatch, tmp_path, timed):
    import app as web
    source = tmp_path / "source.mp4"
    source.touch()
    monkeypatch.setattr(web, "DOWNLOADS_DIR", tmp_path)
    monkeypatch.setattr(web.VideoProcessor, "get_video_info", lambda *a: {"duration": 3})
    monkeypatch.setattr(web.VideoTranscriber, "transcribe", lambda *a, **k: timed)
    recognized = copy.deepcopy(timed)
    recognized["segments"][0]["text"] = "MiMo text"
    provider = MagicMock(return_value=recognized)
    monkeypatch.setattr(mimo.MimoTranscriber, "transcribe_segments", provider)
    analyze = MagicMock(return_value=[])
    monkeypatch.setattr(web.ViralMomentAnalyzer, "analyze_transcript", analyze)
    client = web.app.test_client()
    response = client.post("/api/process", json={"source": "local", "local_file": source.name,
        "provider": "ollama", "transcription_engine": "mimo"})
    events = response.data.decode()
    provider.assert_called_once()
    assert analyze.call_args.args[0] == recognized
    assert "event: notice" in events
    assert client.post("/api/process", json={"transcription_engine": "invalid"}).status_code == 400
    page = client.get("/").data.decode()
    assert 'Xiaomi MiMo API (mimo-v2.5-asr)' in page
    assert "transcription_engine: $('#transcription-engine').value" in page


def test_later_segment_failure_discards_partial_mimo_text(api, timed):
    timed["segments"].append(copy.deepcopy(timed["segments"][0]))
    success = api[1].chat.completions.create.return_value
    api[1].chat.completions.create.side_effect = [success, RuntimeError("private")]
    local = MagicMock()
    local.transcribe.return_value = timed
    result = mimo.transcribe_with_engine(local, "video", "mimo")
    assert result is timed
    assert [s["text"] for s in result["segments"]] == ["Hello", "Hello"]


def test_local_timing_failure_prevents_upload(api):
    local = MagicMock()
    local.transcribe.side_effect = RuntimeError("Model unavailable")
    with pytest.raises(RuntimeError, match="requires local Whisper timing"):
        mimo.transcribe_with_engine(local, "video", "mimo")
    api[0].assert_not_called()


def test_unsupported_language_prevents_upload(api, timed):
    with pytest.raises(mimo.MimoError, match="language hints"):
        mimo.MimoTranscriber().transcribe_segments("video", timed, language="id")
    api[0].assert_not_called()
