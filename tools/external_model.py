"""Locked model identity and persistent console path shared by packaging and controls."""
import re

MODEL_BYTES = 619831552
MODEL_SHA = 'c42ff3ebdfc0b8a0aa0c35a499ccf81303fb9fbdc1a0798c096a098df85003aa'
DEFAULT_MODEL_PATH = '/user/homebrew/ps5-hip/models/tinyllama-q4_0.gguf'


def validate_model_path(path):
    if (not isinstance(path, str) or not 1 <= len(path) <= 240 or
            not path.startswith(('/user/homebrew/', '/data/')) or
            not path.endswith('.gguf') or not re.fullmatch(r'[A-Za-z0-9_./-]+', path) or
            any(part in ('', '.', '..') for part in path[1:].split('/'))):
        raise ValueError('model path must be a bounded persistent .gguf path under /user/homebrew or /data')
    return path


def validate_external_model(spec):
    if (not isinstance(spec, dict) or set(spec) != {'path', 'bytes', 'sha256'} or
            type(spec['bytes']) is not int or spec['bytes'] != MODEL_BYTES or spec['sha256'] != MODEL_SHA):
        raise ValueError('external model identity differs from locked model')
    validate_model_path(spec['path'])
    return spec


def encode_model_path(path):
    return validate_model_path(path).encode('ascii')
