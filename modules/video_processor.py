import ffmpeg
from pathlib import Path
from typing import List, Dict, Optional

from config import OUTPUTS_DIR, AUDIO_CODEC, CLIP_BUFFER_SECONDS
from modules.render_engine import run_render, validate_render_engine
from utils.video_metadata import get_video_info as load_video_info

# Output canvas for vertical clips
VERTICAL_WIDTH = 1080
VERTICAL_HEIGHT = 1920

LAYOUTS = ("center-crop", "split-stack", "blur-background")

# Vertical bias for split-stack panel crops: faces usually sit in the upper
# third of a podcast frame, so the crop window starts above center.
SPLIT_STACK_FACE_BIAS = 0.33


def compute_split_stack_regions(width: int, height: int,
                                face_bias: float = SPLIT_STACK_FACE_BIAS) -> List[Dict]:
    """Compute the two crop regions used to stack side-by-side speakers vertically.

    The source frame is split into left/right halves and each half is cropped
    to a 9:8 panel so the two panels stack into a 9:16 canvas.
    Returns [{'x', 'y', 'width', 'height'}, ...] for the left and right speaker.
    """
    half_width = width // 2
    panel_aspect = (VERTICAL_WIDTH / (VERTICAL_HEIGHT // 2))  # 9:8

    crop_width = half_width
    crop_height = int(crop_width / panel_aspect)
    if crop_height > height:
        crop_height = height
        crop_width = int(crop_height * panel_aspect)

    # Keep ffmpeg happy: crop dimensions must be even
    crop_width -= crop_width % 2
    crop_height -= crop_height % 2

    x_inset = (half_width - crop_width) // 2
    y_offset = int((height - crop_height) * face_bias)

    return [
        {'x': x_inset, 'y': y_offset, 'width': crop_width, 'height': crop_height},
        {'x': half_width + x_inset, 'y': y_offset, 'width': crop_width, 'height': crop_height},
    ]


class VideoProcessor:
    def __init__(self, output_dir: Path = OUTPUTS_DIR, render_engine: str = "auto"):
        self.render_engine = validate_render_engine(render_engine)
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_clip(self, video_path: str, start_time: float, end_time: float,
                    output_name: Optional[str] = None, vertical_format: bool = True,
                    layout: str = "center-crop") -> str:
        if layout not in LAYOUTS:
            raise ValueError(f"Unknown layout '{layout}'. Options: {', '.join(LAYOUTS)}")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        duration = end_time - start_time

        if output_name:
            output_filename = f"{output_name}.mp4"
        else:
            output_filename = f"{video_path.stem}_clip_{int(start_time)}_{int(end_time)}.mp4"

        output_path = self.output_dir / output_filename

        try:
            # Extract clip segment

            input_stream = ffmpeg.input(str(video_path), ss=start_time, t=duration)

            if vertical_format:
                # Probe once per source file and reuse cached metadata across clips.
                video_info = load_video_info(str(video_path))
                width = int(video_info['width'])
                height = int(video_info['height'])

                audio = input_stream.audio

                if layout == "blur-background":
                    # Blur on a smaller canvas, then upscale: one FFmpeg graph,
                    # with the complete source fitted over the centered background.
                    split = input_stream.video.filter_multi_output('split')
                    background = (split[0]
                        .filter('scale', 270, 480, force_original_aspect_ratio='increase')
                        .filter('crop', 270, 480)
                        .filter('setsar', 1)
                        .filter('gblur', sigma=12)
                        .filter('scale', VERTICAL_WIDTH, VERTICAL_HEIGHT))
                    foreground = (split[1]
                        .filter('scale', VERTICAL_WIDTH, VERTICAL_HEIGHT,
                                force_original_aspect_ratio='decrease', force_divisible_by=2)
                        .filter('setsar', 1))
                    video = ffmpeg.overlay(background, foreground,
                                           x='(W-w)/2', y='(H-h)/2')
                elif layout == "split-stack":
                    # Two side-by-side speakers (podcast framing): crop each
                    # half and stack them vertically into the 9:16 canvas.
                    regions = compute_split_stack_regions(width, height)
                    split = input_stream.video.filter_multi_output('split')
                    panels = []
                    for index, region in enumerate(regions):
                        panel = ffmpeg.filter(
                            split[index], 'crop',
                            region['width'], region['height'],
                            region['x'], region['y']
                        )
                        panel = ffmpeg.filter(
                            panel, 'scale',
                            VERTICAL_WIDTH, VERTICAL_HEIGHT // 2
                        )
                        panels.append(panel)
                    video = ffmpeg.filter(panels, 'vstack', inputs=2)
                else:
                    # Calculate 9:16 crop (vertical format for social media)
                    target_aspect = 9 / 16
                    current_aspect = width / height

                    if current_aspect > target_aspect:
                        # Video is wider than 9:16, crop width
                        new_width = int(height * target_aspect)
                        new_height = height
                        x_offset = (width - new_width) // 2
                        y_offset = 0
                    else:
                        # Video is taller than 9:16, crop height
                        new_width = width
                        new_height = int(width / target_aspect)
                        x_offset = 0
                        y_offset = (height - new_height) // 2

                    video = ffmpeg.filter(input_stream.video, 'crop', new_width, new_height, x_offset, y_offset)
                    video = ffmpeg.filter(video, 'scale', VERTICAL_WIDTH, VERTICAL_HEIGHT)

            else:
                video = input_stream.video

            # Keep the CPU graph contiguous; NVENC uploads only its final frames.
            def build_stream(options):
                return ffmpeg.output(
                    video, input_stream['a?'], str(output_path),
                    acodec=AUDIO_CODEC, movflags='faststart',
                    **{'b:a': '128k'}, **options)

            self.last_encoder = run_render(build_stream, self.render_engine, quiet=True)

            if not output_path.exists():
                raise Exception("Output file was not created")
            
            # Clip saved
            return str(output_path)
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise Exception(f"FFmpeg error: {error_msg}")
        except Exception as e:
            raise Exception(f"Error extracting clip: {str(e)}")
    
    def extract_multiple_clips(self, video_path: str, moments: List[Dict],
                             prefix: str = "viral_clip", vertical_format: bool = True,
                             layout: str = "center-crop") -> List[str]:
        output_paths = []

        for i, moment in enumerate(moments):
            try:
                output_name = f"{prefix}_{i+1}_score_{moment['score']:.1f}"
                output_path = self.extract_clip(
                    video_path,
                    moment['start'],
                    moment['end'],
                    output_name,
                    vertical_format=vertical_format,
                    layout=layout
                )
                output_paths.append(output_path)
                
                metadata_path = Path(output_path).with_suffix('.json')
                import json
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'original_video': str(video_path),
                        'start_time': moment['start'],
                        'end_time': moment['end'],
                        'duration': moment['duration'],
                        'score': moment['score'],
                        'reason': moment['reason'],
                        'text_preview': moment.get('text', '')
                    }, f, indent=2)
                    
            except Exception as e:
                # Failed to extract clip
                continue
        
        return output_paths
    
    def get_video_info(self, video_path: str) -> Dict:
        try:
            return load_video_info(video_path)
        except Exception as e:
            raise Exception(f"Error getting video info: {str(e)}")
    
    def validate_timestamps(self, video_path: str, moments: List[Dict]) -> List[Dict]:
        video_info = self.get_video_info(video_path)
        video_duration = video_info['duration']
        
        valid_moments = []
        for moment in moments:
            if moment['start'] < 0:
                moment['start'] = 0
            if moment['end'] > video_duration:
                moment['end'] = video_duration
            
            if moment['start'] < moment['end']:
                moment['duration'] = moment['end'] - moment['start']
                valid_moments.append(moment)
            else:
                # Skip invalid moment
                pass
        
        return valid_moments


if __name__ == "__main__":
    processor = VideoProcessor()
    
    video_path = input("Enter video file path: ")
    start_time = float(input("Enter start time (seconds): "))
    end_time = float(input("Enter end time (seconds): "))
    
    try:
        output_path = processor.extract_clip(video_path, start_time, end_time)
        output_path = processor.extract_clip(video_path, start_time, end_time)
    except Exception as e:
        pass
