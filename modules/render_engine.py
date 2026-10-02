"""Verified hardware encoding; CPU filters stay in system memory until NVENC."""
import logging
import shutil
import subprocess
import threading
import time

import ffmpeg

ENGINES = ('auto', 'nvenc', 'cpu')
_lock = threading.Lock()
_cache = {}
log = logging.getLogger(__name__)


def validate_render_engine(value):
    if value not in ENGINES:
        raise ValueError('Render engine must be auto, nvenc, or cpu')
    return value


def encoder_options(encoder, cpu_preset='medium'):
    if encoder == 'h264_nvenc':
        return dict(vcodec=encoder, preset='p4', tune='hq', rc='vbr', cq=23,
                    **{'b:v': 0}, pix_fmt='yuv420p')
    return dict(vcodec='libx264', preset=cpu_preset, crf=23, pix_fmt='yuv420p')


def capabilities():
    executable = shutil.which('ffmpeg') or 'ffmpeg'
    with _lock:
        cached = _cache.get(executable)
        if cached and time.monotonic() - cached[0] < 300:
            return dict(cached[1])
        # A listed encoder is insufficient: open a real NVENC session and encode.
        command = ffmpeg.compile(ffmpeg.output(
            ffmpeg.input('color=size=640x360:rate=30', f='lavfi'), '-',
            frames=3, f='null', **encoder_options('h264_nvenc')), cmd=executable)
        try:
            result = subprocess.run(command, capture_output=True, timeout=15,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            available = result.returncode == 0
            reason = '' if available else result.stderr.decode(errors='replace')[-1500:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            available, reason = False, str(exc)
        info = dict(ffmpeg=executable, nvenc=available,
                    auto_encoder='h264_nvenc' if available else 'libx264', reason=reason)
        _cache[executable] = (time.monotonic(), info)
        return dict(info)


def run_render(build_stream, engine='auto', cpu_preset='medium', **run_options):
    """Rebuild the same graph with CPU options if Auto's hardware encode fails."""
    validate_render_engine(engine)
    encoder = 'libx264'
    if engine != 'cpu':
        info = capabilities()
        if info['nvenc']:
            encoder = 'h264_nvenc'
        elif engine == 'nvenc':
            raise RuntimeError('NVIDIA NVENC is unavailable. Select Auto or CPU. ' + info['reason'])
    log.info('Render engine %s: %s (CPU decode/filters)', engine, encoder)
    try:
        ffmpeg.run(build_stream(encoder_options(encoder, cpu_preset)),
                   overwrite_output=True, **run_options)
    except ffmpeg.Error:
        if engine != 'auto' or encoder != 'h264_nvenc':
            raise
        log.warning('NVENC render failed; retrying the complete render with libx264')
        ffmpeg.run(build_stream(encoder_options('libx264', cpu_preset)),
                   overwrite_output=True, **run_options)
        encoder = 'libx264'
    return encoder
