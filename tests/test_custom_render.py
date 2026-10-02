import copy
from pathlib import Path
import subprocess

import ffmpeg
import pytest

from config import SUBTITLE_TEMPLATES
from modules.subtitle_generator import SubtitleGenerator
from modules.subtitle_settings import validate_subtitle_settings
from modules.video_processor import VideoProcessor


@pytest.mark.parametrize('settings', [
    {'font': 'Arial\nStyle: injected'}, {'color': 'red'}, {'outline': -1},
    {'fontsize': float('nan')}, {'alignment': 2.5}, {'unexpected': 1}, [],
])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        validate_subtitle_settings(settings)


def test_ass_custom_style(tmp_path):
    generator = SubtitleGenerator(tmp_path)
    style = {**SUBTITLE_TEMPLATES['TikTok Style']['vertical'],
             **validate_subtitle_settings({'font': 'Arial', 'fontsize': 72,
                 'color': '#123456', 'stroke_color': '#abcdef', 'outline': 0,
                 'position': 0.15, 'alignment': 4})}
    ass = Path(generator._create_ass_file(
        [{'text': 'Custom subtitle', 'start': 0, 'end': 1}], style, 0))
    try:
        text = ass.read_text(encoding='utf-8')
        assert 'Style: Default,Arial,72,&H00563412' in text
        assert '&H00EFCDAB' in text
        assert r'\pos(32,288)' in text
        assert ',1,0.0,0,4,' in text
    finally:
        ass.unlink()


@pytest.mark.parametrize('with_audio', [True, False])
def test_real_blur_and_subtitle_render(tmp_path, with_audio):
    source = tmp_path / 'horizontal source.mp4'
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
               'testsrc2=size=640x360:rate=12:duration=1']
    if with_audio:
        command += ['-f', 'lavfi', '-i', 'sine=frequency=440:duration=1']
    subprocess.run(command + ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
    video = VideoProcessor(tmp_path).extract_clip(str(source), 0, 1,
                                                 layout='blur-background')
    transcript = {'language': 'en', 'segments': [{'start': 0, 'end': 1, 'words': [
        {'word': 'Hello', 'start': 0.1, 'end': 0.5},
        {'word': 'world', 'start': 0.5, 'end': 0.9}]}]}
    original = copy.deepcopy(SUBTITLE_TEMPLATES)
    output = SubtitleGenerator(tmp_path).add_subtitles(video, transcript, 0, 1,
        style_template='TikTok Style', subtitle_settings={
            'font': 'Arial', 'fontsize': 100, 'color': '#00ff00',
            'outline': 4, 'position': 0.8, 'alignment': 5})
    streams = ffmpeg.probe(output)['streams']
    rendered = next(s for s in streams if s['codec_type'] == 'video')
    assert (rendered['width'], rendered['height']) == (1080, 1920)
    assert rendered['sample_aspect_ratio'] == '1:1'
    assert any(s['codec_type'] == 'audio' for s in streams) == with_audio
    assert SUBTITLE_TEMPLATES == original


def test_api_validation():
    from app import app
    client = app.test_client()
    assert client.get('/').status_code == 200
    assert 'TikTok Style' in client.get('/api/subtitle-styles').json
    for payload in ({'layout': 'bad'}, {'subtitle_settings': {'outline': -1}}):
        response = client.post('/api/process', json=payload)
        assert response.status_code == 400


def test_process_and_restyle_forward_settings(tmp_path, monkeypatch):
    import json
    import app as web
    source = tmp_path / 'source.mp4'
    source.touch()
    clip = tmp_path / 'clip.mp4'
    clip.touch()
    monkeypatch.setattr(web, 'DOWNLOADS_DIR', tmp_path)
    monkeypatch.setattr(web.VideoProcessor, 'get_video_info', lambda *a: {'duration': 1})
    monkeypatch.setattr(web.VideoTranscriber, 'transcribe', lambda *a, **k: {'language': 'en'})
    moment = {'start': 0, 'end': 1, 'duration': 1, 'score': 9}
    monkeypatch.setattr(web.ViralMomentAnalyzer, 'analyze_transcript', lambda *a, **k: [moment])
    monkeypatch.setattr(web.ViralMomentAnalyzer, 'refine_moments', lambda *a: [moment])
    monkeypatch.setattr(web.ViralMomentAnalyzer, 'generate_clip_metadata', lambda *a, **k: None)
    monkeypatch.setattr(web.VideoProcessor, 'validate_timestamps', lambda *a: [moment])
    calls = []
    def render(*args, **kwargs):
        kwargs['render_engine'] = args[0].render_engine
        calls.append(kwargs)
        return str(clip)
    monkeypatch.setattr(web.VideoProcessor, 'extract_clip', render)
    monkeypatch.setattr(web.SubtitleGenerator, 'add_subtitles', render)
    client = web.app.test_client()
    settings = {'font': 'Arial', 'color': '#00ff00', 'outline': 4,
                'animation': 'bounce', 'active_word': True}
    response = client.post('/api/process', json={
        'source': 'local', 'local_file': source.name, 'provider': 'ollama',
        'layout': 'blur-background', 'subtitle_style': 'TikTok Style',
        'subtitle_settings': settings, 'render_engine': 'nvenc'})
    events = response.data.decode()
    assert 'event: error' not in events
    done = json.loads(events.split('event: done\ndata: ')[1].split('\n')[0])
    assert calls[0]['layout'] == 'blur-background'
    assert calls[1]['subtitle_settings'] == settings
    assert calls[1]['style_template'] == 'TikTok Style'
    assert calls[0]['render_engine'] == calls[1]['render_engine'] == 'nvenc'
    try:
        response = client.post('/api/restyle', json={'session_id': done['session_id'],
            'clip_index': 0, 'style': 'Classic', 'subtitle_settings': settings})
        assert response.status_code == 200
        assert calls[2]['subtitle_settings'] == settings
        response = client.post('/api/restyle', json={'session_id': done['session_id'],
            'clip_index': 0, 'style': 'Classic'})
        assert response.status_code == 200
        assert calls[3]['subtitle_settings'] == settings
        assert calls[2]['render_engine'] == calls[3]['render_engine'] == 'nvenc'
    finally:
        web._sessions.pop(done['session_id'], None)

