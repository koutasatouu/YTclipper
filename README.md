# YouTube to Viral Clips

Submagic free open source alternative that can run with local AI model using Ollama and OpenAI Whisper (no API key needed).

## Overview

This tool downloads YouTube videos, transcribes them using OpenAI Whisper, analyzes the content for viral potential using AI (Ollama/OpenAI/Anthropic), and automatically extracts the most engaging clips with optional subtitles.

## Features

- Automatic viral moment detection with customizable scoring threshold
- Multi-language support (transcription and subtitles)
- Parallel clip extraction for faster processing
- Multiple AI providers: Ollama (local), OpenAI, Anthropic
- Customizable subtitle styles optimized for social media
- Karaoke word highlighting: the actively spoken word pops in color ("Viral Highlight" style)
- Vertical format option for TikTok/Reels/Shorts
- Split-stack layout for two-person podcasts: side-by-side speakers are stacked vertically
- Clean, minimal web interface

## Requirements

- Python 3.8+
- FFmpeg
- Ollama (for local AI) or API keys for OpenAI/Anthropic

## Installation

1. Clone the repository:
```bash
git clone https://github.com/guillaumegay13/youtube-to-viral-clips.git
cd youtube-to-viral-clips
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

4. Set up AI provider:

For Ollama (local, free):
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull qwen3.5:9b-q4_K_M
ollama create qwen3.5-9b-q4km -f Modelfile.qwen3.5

# Start Ollama server
ollama serve
```

For OpenAI:
```bash
export OPENAI_API_KEY="your-api-key"
```

For Anthropic:
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

## Usage

### Web Interface

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

### Command Line

For direct Python usage:
```python
from modules.downloader import YouTubeDownloader
from modules.transcriber import VideoTranscriber
from modules.analyzer import ViralMomentAnalyzer
from modules.video_processor import VideoProcessor

# Download video
downloader = YouTubeDownloader()
video_data = downloader.download("https://youtube.com/watch?v=...", "720p")

# Transcribe
transcriber = VideoTranscriber()
transcript = transcriber.transcribe(video_data['filepath'])

# Analyze for viral moments
analyzer = ViralMomentAnalyzer(provider="ollama")
viral_moments = analyzer.analyze_transcript(transcript)

# Extract clips
processor = VideoProcessor()
for moment in viral_moments[:3]:
    processor.extract_clip(
        video_data['filepath'],
        moment['start'],
        moment['end'],
        f"clip_score_{moment['score']:.1f}"
    )
```

## Configuration

Edit `config.py` to customize:

- `AI_PROVIDER`: Choose between "ollama", "openai", or "anthropic"
- `MIN_VIRAL_SCORE`: Minimum score threshold (0-10)
- `MIN_CLIP_LENGTH`: Minimum clip duration in seconds
- `MAX_CLIP_LENGTH`: Maximum clip duration in seconds
- `WHISPER_MODEL`: Whisper model ("base", "small", "medium", "large-v3", "large-v3-turbo"). Default is "large-v3-turbo" — smaller models produce poor subtitles on non-English audio. Override per run with `--whisper-model`; pass `--language fr` to skip auto-detection.

## Podcast Mode (Split-Stack Layout)

For two-person podcasts filmed with both speakers side by side (16:9), use the
`split-stack` layout to convert to shorts: each speaker's half of the frame is
cropped and the two are stacked vertically into the 9:16 canvas.

```bash
python main.py --file "episode.mp4" --layout split-stack --subtitle-style "Viral Highlight"
```

The default `center-crop` layout keeps the middle of the frame and is right for
single-speaker videos.

## Subtitle Styles

Available subtitle templates:
- Classic: White text with black outline
- Bold Yellow: Yellow text with thick black outline
- Submagic Yellow: Bold yellow text with smart emoji placement
- Minimal: Small white text with thin outline
- TikTok Style: Large white text with colored shadow
- Neon: Cyan text with purple glow
- Ultra Bold: Extra thick white text
- Viral Bold: Massive white text for maximum impact
- Viral Highlight: Karaoke-style word-by-word highlight — the spoken word pops in yellow

## Output

Clips are saved to the `outputs/` directory with the following naming:
- `clip_1_score_8.5.mp4` (without subtitles)
- `clip_1_final.mp4` (with subtitles)

## API Keys

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

## Troubleshooting

### "Missing dependencies" error
Ensure FFmpeg is installed and accessible in your PATH.

### "Ollama connection failed"
Start the Ollama server with `ollama serve`.

### "No viral moments found"
Try lowering the minimum score threshold or using a different video.

### Subtitle language issues
The tool automatically detects the video language. Ensure Whisper's transcription task is set to "transcribe" (not "translate") in config.py.

## License

MIT License

## Contributing

Pull requests welcome. For major changes, please open an issue first.

## Custom subtitles and Blur Background

In the web UI, open **Processing options**:

1. Keep **Vertical format (9:16)** enabled.
2. Choose **Video Layout > Blur Background** to fit the complete video at the center of a 1080x1920 blurred duplicate. **Auto Crop (default)** preserves the existing center-crop behavior. Split Stack remains available.
3. Enable **Add subtitles** and choose **TikTok Style** or another existing preset.
4. Open **Customize subtitle preset**, enable **Use custom settings**, then adjust font family, size, text/outline colors, outline width, position, and alignment. Sizes are relative to a 1920px-high video. Use an installed font; libass falls back if unavailable.
5. Click **Extract Viral Clips**. Choosing a different preset disables custom overrides; enable them again to customize that preset. The result-card restyle menu applies its chosen preset with its own defaults.

The existing ASS subtitle system, grouping, and highlighted presets are retained. Blur is computed at 270x480 and upscaled inside a single FFmpeg filter graph; no Python frame processing is used. The clean intermediate clip remains available for restyling.

CLI layout: `python main.py --url "URL" --layout blur-background --subtitle-style "TikTok Style"`.

Text animation: choose **None**, **Pop** (80% → 108% → 100%, up to 180ms),
or **Bounce** (65% → 122% → 94% → 100%, up to 300ms). Choosing TikTok Style
sets Pop automatically. Enable **Animate active word only** to animate each spoken
word using transcription timestamps; otherwise the caption animates as a group.
Short captions shorten the animation. Without word timestamps, animation falls back
to the caption. All animation uses native ASS/libass transforms, no Python frame loop.
Rendering preferences are saved in this browser and restored after reload. Choosing
a new preset disables custom font/color overrides. Restyle retains the original
session's overrides unless `subtitle_settings` is explicitly supplied (use `{}` to reset).

CLI example: `python main.py --file "episode.mp4" --layout blur-background --subtitle-style "TikTok Style" --text-animation pop --active-word`.

No-final-video verification: `python -m pytest tests/ -q -k "not real_blur_and_subtitle_render"`.
This includes short synthetic FFmpeg checks to the null sink; it does not create final videos.

API: `/api/process` accepts `layout` (`center-crop`, `blur-background`, `split-stack`) and optional `subtitle_settings`. Both `/api/process` and `/api/restyle` accept `animation` (`none`, `pop`, `bounce`), `active_word` (boolean), and overrides such as:

```json
{"font":"Arial","fontsize":100,"color":"#ffffff","stroke_color":"#000000","outline":4,"position":0.8,"alignment":5}
```

Position is a fraction from the top (0.05–0.95), alignment uses ASS values 1–9, and colors use #RRGGBB. Invalid overrides return HTTP 400. Omitting overrides preserves preset rendering.

Validation: `python -m pytest tests/ -q` includes actual FFmpeg renders with and without audio, ASS styling checks, and process/restyle API integration with transcription/analysis stubbed. These tests do not call an AI service or download a YouTube video.

### GPU render engine

In **Processing options → Render Engine**, choose **Auto (Recommended)**,
**NVIDIA NVENC**, or **CPU (libx264)**. Preferences persist in this browser.
CLI: `--render-engine auto|nvenc|cpu`. API: `render_engine` on `/api/process`
and `/api/restyle`; restyling defaults to the original session's engine.

Auto runs a short synthetic H.264 NVENC encode using FFmpeg from PATH, caches
its result for five minutes, and selects NVIDIA only when the test succeeds.
If the actual NVENC render fails, Auto rebuilds the same graph with libx264
and retries the complete clip. Explicit NVIDIA reports an error if unavailable;
CPU bypasses hardware probing. `/api/render-capabilities` reports the probe result.
No GPU model is hardcoded. Both clip extraction and final subtitle encoding use
the selected engine. NVIDIA uses p4/hq, VBR/CQ 23; CPU retains its previous
medium (extraction) and veryfast (subtitles) presets with CRF 23. These quality
settings are not guaranteed to produce identical bitrate or visual quality.

ASS/libass, Pop/Bounce, crop/scale, split-stack, and blur/overlay remain CPU filters.
The existing blur runs at 270×480 before upscaling. Frames stay in CPU memory
through each filter graph and upload once at NVENC encoding. NVDEC/CUDA filters
are deliberately not enabled for this CPU-dependent pipeline, avoiding an extra
GPU-to-CPU download. The pre-subtitle clip remains available for restyling, so
extraction and subtitle burning remain separate passes. Audio behavior is unchanged.

Validation uses short synthetic clips only: `python -m pytest tests/ -q`.
Hardware integration tests skip when the NVENC capability test fails.


### Xiaomi MiMo speech recognition

In **Processing options → Speech recognition → Transcription engine**, choose
**Whisper Local** (default) or **Xiaomi MiMo API (mimo-v2.5-asr)**.
Set `MIMO_API_KEY` in the server environment (the existing project dotenv loader
also supports this variable), then restart the app. Never put API keys in the UI.
The existing `openai`, `python-dotenv` and FFmpeg dependencies are sufficient.
CLI: `python main.py --file video.mp4 --transcription-engine mimo`.

MiMo uses the official `https://api.xiaomimimo.com/v1/chat/completions` endpoint.
Local Whisper supplies speech segment boundaries; each segment is extracted as
16 kHz mono WAV and sent as Base64 (maximum 10 MB per request). Segments longer
than 180 seconds fall back safely. One API request is made per speech segment;
MiMo mode still runs Whisper and does not reduce local transcription work.
MiMo text feeds clip analysis. Subtitle words and their genuine timestamps remain
from Whisper, including restyling; they may differ from MiMo's wording. No word
timestamps are fabricated. Whisper transcript caches are never overwritten by
MiMo. MiMo requests are not cached, so rerunning MiMo may incur API usage again.
Missing keys, API failures, or incomplete responses produce a visible notice and
fall back to the complete Whisper transcript. An unavailable local Whisper model
is an error: this pipeline requires local timing even in MiMo mode.
The documented MiMo language hints are `auto`, `en`, and `zh`; use Whisper for
other languages. Official API reference:
https://mimo.mi.com/docs/en-US/api/audio/Speech-Recognition


### Local multilingual clip analysis

The default Ollama model is `qwen3.5-9b-q4km`, based on the explicit official
`qwen3.5:9b-q4_K_M` tag. Set `LLM_MODEL` in `.env` or the environment to change it.
`OLLAMA_NUM_CTX=8192`, `OLLAMA_NUM_PREDICT=1024`, and `OLLAMA_THINK=false` are the
conservative analyzer defaults for an 8 GB GPU. Local inference runs sequentially.

The analyzer evaluates every transcript window, including Indonesian, English,
other languages and mixed-language speech. Optional keyword ranking only changes
evaluation order; it never skips windows. Prompts evaluate setup, build-up, payoff,
reaction and multilingual context. Titles/descriptions use the transcript language.
The default `ANALYSIS_PROFILE=gaming` emphasizes reactions, surprise, comedy,
skill and payoff, with explicit filler penalties. Set `ANALYSIS_PROFILE=general`
to retain the previous story/insight/quotability rubric and generic categories.
Analysis caches include the model, prompt hash, analyzer version, language and
inference settings. Failed/partial inference is not cached as successful analysis.

Gaming mode uses 16 event categories, a deterministic 0–10 scoring rubric,
validated peak segment + setup/aftermath boundaries, and duplicate suppression.
Complete short reactions may be 3–60 seconds rather than padded to 15 seconds.
No below-threshold top-three fallback is used in gaming mode. It can return no
clips; lowering the UI minimum does not bypass the analyzer floor (6 by default).
The analyzer still reads transcripts, not raw audio/video or audience chat.
See [gaming research, category guide and scoring parameters](GAMING_SELECTION.md).
