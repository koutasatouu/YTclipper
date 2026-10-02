"""No final videos: validate ASS and actual pipeline graphs against FFmpeg's null sink."""
from pathlib import Path
import re
import subprocess

import ffmpeg
import pytest

from config import SUBTITLE_TEMPLATES
from modules.subtitle_generator import SubtitleGenerator
from modules.subtitle_settings import validate_subtitle_settings


@pytest.mark.parametrize('value', ['spin', None, 3, [], {}])
def test_reject_invalid_animation(value):
    with pytest.raises(ValueError):
        validate_subtitle_settings({'animation': value})


@pytest.mark.parametrize('mode,peak', [('none', None), ('pop', 108), ('bounce', 122)])
@pytest.mark.parametrize('active', [False, True])
def test_ass_animation(mode, peak, active, tmp_path):
    generator = SubtitleGenerator(tmp_path)
    words = [{'word': 'Hello', 'start': 0, 'end': .4},
             {'word': 'world', 'start': .4, 'end': .8}]
    style = {**SUBTITLE_TEMPLATES['TikTok Style']['vertical'],
             'animation': mode, 'active_word': active}
    path = Path(generator._create_ass_file(
        [{'text': 'Hello world', 'start': 0, 'end': .8, 'words': words}], style, 0))
    try:
        text = path.read_text(encoding='utf-8')
        assert (r'\t(' in text) == (mode != 'none')
        if peak:
            assert rf'\fscx{peak}' in text
        if active and peak:
            for line in text.splitlines():
                if line.startswith('Dialogue:'):
                    assert line.count(r'\t(') == (2 if mode == 'pop' else 3)
        for start, end in re.findall(r'\\t\((-?\d+),(-?\d+),', text):
            assert int(start) < int(end) <= 300
    finally:
        path.unlink()


def test_short_animation_and_continuation():
    tag = SubtitleGenerator._animation_tag('pop', .04)
    assert r'\t(18,40,\fscx100\fscy100)' in tag
    assert SubtitleGenerator._animation_tag('pop', 1, .4) == r'{\fscx100\fscy100}'


def test_ui_payload_builder():
    import json
    html = (Path(__file__).parents[1] / 'templates' / 'index.html').read_text(encoding='utf-8')
    function = html.split('    function customSubtitleSettings() {', 1)[1].split('\n    }', 1)[0]
    controls = {'text-animation': {'value': 'bounce'}, 'active-word': {'checked': True},
                'custom-subtitles': {'checked': True}, 'subtitle-font': {'value': ' Arial '},
                'subtitle-size': {'value': '72'}, 'subtitle-color': {'value': '#123456'},
                'subtitle-outline-color': {'value': '#000000'}, 'subtitle-outline': {'value': '4'},
                'subtitle-position': {'value': '0.8'}, 'subtitle-alignment': {'value': '5'}}
    script = 'const controls = ' + json.dumps(controls) + '; const $ = id => controls[id.slice(1)];\n'
    script += 'function customSubtitleSettings() {' + function + '\n}\n'
    script += "console.log(JSON.stringify(customSubtitleSettings())); controls['custom-subtitles'].checked = false; console.log(JSON.stringify(customSubtitleSettings()));"
    result = subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)
    custom, preset = [json.loads(line) for line in result.stdout.splitlines()]
    assert custom == {'font': 'Arial', 'fontsize': 72, 'color': '#123456', 'stroke_color': '#000000',
                      'outline': 4, 'position': .8, 'alignment': 5, 'animation': 'bounce', 'active_word': True}
    assert preset == {'animation': 'bounce', 'active_word': True}


@pytest.mark.parametrize('layout', ['center-crop', 'blur-background', 'split-stack'])
def test_actual_layout_and_subtitle_graph_to_null(layout, tmp_path, monkeypatch):
    import modules.video_processor as processor
    import modules.subtitle_generator as subtitles
    source = tmp_path / 'horizontal source.mp4'
    source.touch()  # FFmpeg input is replaced by a synthetic source below.
    output = tmp_path / 'validated.mp4'
    output.touch()  # Satisfy the production existence check; no video is written.
    monkeypatch.setattr(processor, 'load_video_info', lambda *a: {'width': 640, 'height': 360})
    commands = []
    def run_null(stream, **kwargs):
        command = ffmpeg.compile(stream)
        idx = command.index('-i')
        command[idx:idx+2] = ['-f', 'lavfi', '-i', 'testsrc2=size=640x360:rate=12:duration=0.5']
        command[-1:] = ['-c:v', 'wrapped_avframe', '-an', '-frames:v', '4', '-f', 'null', '-']
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        commands.append(command)
    monkeypatch.setattr(ffmpeg, 'run', run_null)
    processor.VideoProcessor(tmp_path).extract_clip(str(source), 0, .5, 'validated', layout=layout)
    graph = commands[0][commands[0].index('-filter_complex') + 1]
    if layout == 'blur-background':
        assert 'gblur' in graph and 'overlay' in graph and 'split' in graph
    monkeypatch.setattr(subtitles, 'get_video_info', lambda *a: {'duration': .5, 'width': 640, 'height': 360})
    generator = SubtitleGenerator(tmp_path)
    monkeypatch.setattr(generator.transcriber, 'get_words_in_range', lambda *a: [
        {'word': 'Hello', 'start': 0, 'end': .25}, {'word': 'world', 'start': .25, 'end': .5}])
    for animation in ('none', 'pop', 'bounce'):
        generator.add_subtitles(str(source), {}, 0, .5, subtitle_settings={
            'font': 'Arial', 'animation': animation, 'active_word': True})
    assert len(commands) == 4
