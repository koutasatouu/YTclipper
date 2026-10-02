"""Validated overrides shared by the web API and ASS renderer."""
import math
import re


def validate_subtitle_settings(settings):
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise ValueError('Subtitle settings must be an object')
    result = {}
    bounds = {'fontsize': (20, 240), 'outline': (0, 20),
              'position': (0.05, 0.95), 'alignment': (1, 9)}
    for key, value in settings.items():
        if key == 'animation':
            if value not in ('none', 'pop', 'bounce'):
                raise ValueError('Animation must be none, pop, or bounce')
            result[key] = value
        elif key == 'active_word':
            if not isinstance(value, bool):
                raise ValueError('Active word must be a boolean')
            result[key] = value
        elif key == 'font':
            if not isinstance(value, str) or not re.fullmatch(r'[\w .-]{1,80}', value.strip()):
                raise ValueError('Font must be a font family name (1–80 characters)')
            result[key] = value.strip()
        elif key in ('color', 'stroke_color'):
            if not isinstance(value, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', value):
                raise ValueError(f'{key} must be a #RRGGBB color')
            result[key] = value
        elif key in bounds:
            low, high = bounds[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{key} must be between {low} and {high}')
            if key == 'alignment' and int(value) != value:
                raise ValueError('Alignment must be an integer from 1 to 9')
            result[key] = value
        else:
            raise ValueError(f'Unknown subtitle setting: {key}')
    return result
