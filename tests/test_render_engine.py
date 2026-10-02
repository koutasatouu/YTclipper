import subprocess

import ffmpeg
import pytest

from modules import render_engine as engine


@pytest.mark.parametrize('mode,available,expected', [
    ('auto', True, 'h264_nvenc'), ('auto', False, 'libx264'),
    ('cpu', True, 'libx264'), ('nvenc', True, 'h264_nvenc')])
def test_selection(monkeypatch, mode, available, expected):
    monkeypatch.setattr(engine, 'capabilities', lambda: {'nvenc': available, 'reason': 'driver'})
    calls = []
    monkeypatch.setattr(ffmpeg, 'run', lambda stream, **kw: calls.append(stream))
    assert engine.run_render(lambda options: options, mode) == expected
    assert calls[0]['vcodec'] == expected
    assert ('crf' in calls[0]) == (expected == 'libx264')


def test_runtime_fallback_rebuilds_options(monkeypatch):
    monkeypatch.setattr(engine, 'capabilities', lambda: {'nvenc': True})
    calls = []
    def run(options, **kwargs):
        calls.append(options)
        if len(calls) == 1:
            raise ffmpeg.Error('ffmpeg', b'', b'NVENC session unavailable')
    monkeypatch.setattr(ffmpeg, 'run', run)
    assert engine.run_render(lambda options: options) == 'libx264'
    assert [c['vcodec'] for c in calls] == ['h264_nvenc', 'libx264']
    assert 'cq' not in calls[1] and 'rc' not in calls[1]


def test_explicit_nvenc_unavailable(monkeypatch):
    monkeypatch.setattr(engine, 'capabilities', lambda: {'nvenc': False, 'reason': 'driver missing'})
    with pytest.raises(RuntimeError, match='Select Auto or CPU'):
        engine.run_render(lambda options: options, 'nvenc')


@pytest.mark.parametrize('failure', [FileNotFoundError(), subprocess.TimeoutExpired('ffmpeg', 15)])
def test_probe_failure_and_cache(monkeypatch, failure):
    monkeypatch.setattr(engine, '_cache', {})
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise failure
    monkeypatch.setattr(subprocess, 'run', fail)
    assert engine.capabilities()['auto_encoder'] == 'libx264'
    assert not engine.capabilities()['nvenc']
    assert len(calls) == 1


def test_api_rejects_unknown_engine():
    from app import app
    with app.test_client() as client:
        for endpoint in ('/api/process', '/api/restyle'):
            assert client.post(endpoint, json={'render_engine': 'unknown'}).status_code == 400


@pytest.mark.parametrize('mode', ['cpu', 'nvenc'])
def test_synthetic_blur_animation_encoding(tmp_path, mode):
    from modules.video_processor import VideoProcessor
    from modules.subtitle_generator import SubtitleGenerator
    if mode == 'nvenc' and not engine.capabilities()['nvenc']:
        pytest.skip('No working NVIDIA encoder on this host')
    source = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=640x360:rate=12:duration=0.5',
                    '-c:v', 'libx264', str(source)], check=True)
    processor = VideoProcessor(tmp_path, render_engine=mode)
    clip = processor.extract_clip(str(source), 0, .5, layout='blur-background')
    generator = SubtitleGenerator(tmp_path, render_engine=mode)
    transcript = {'segments': [{'start': 0, 'end': .5, 'words': [
        {'word': 'GPU', 'start': 0, 'end': .25},
        {'word': 'test', 'start': .25, 'end': .5}]}]}
    for animation in ('pop', 'bounce'):
        result = generator.add_subtitles(clip, transcript, 0, .5,
            subtitle_settings={'animation': animation, 'active_word': True})
        video = next(s for s in ffmpeg.probe(result)['streams'] if s['codec_type'] == 'video')
        assert (video['width'], video['height']) == (1080, 1920)
        assert video['codec_name'] == 'h264'
        expected = 'h264_nvenc' if mode == 'nvenc' else 'libx264'
        assert processor.last_encoder == generator.last_encoder == expected
