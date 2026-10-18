"""Path and identity checks for frozen full-model artifacts.

The caller supplies an independently trusted manifest digest and the runner's
fixed required-code set. Neither may come solely from the candidate itself.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath

BUDGETS = {'absolute': 2e-5, 'relative': 2e-4, 'relative_l2': 2e-4}
PROFILES = {'smoke': (3, 72), 'generation': (16, 384), 'corpus': (272, 6528)}

def require(value, message):
    if not value:
        raise ValueError(message)

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def load(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs)

def child(root, name):
    require(isinstance(name, str) and name and '\\' not in name, 'relative path required')
    parts = PurePosixPath(name).parts
    require(not PurePosixPath(name).is_absolute() and '/'.join(parts) == name and
            all(p not in ('.', '..') and ':' not in p for p in parts), 'noncanonical path')
    path = root
    for part in parts:
        path = path / part
        require(not path.is_symlink(), 'symlink forbidden')
    require(path.resolve().is_relative_to(root.resolve()), 'path escape')
    return path

def inventory(root, expected):
    require(root.is_dir() and not root.is_symlink(), 'artifact directory')
    paths = list(root.rglob('*'))
    require(not any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths), 'artifact kind')
    names = {p.relative_to(root).as_posix() for p in paths if p.is_file()}
    directories = {p.relative_to(root).as_posix() for p in paths if p.is_dir()}
    required_dirs = {str(parent) for name in expected for parent in PurePosixPath(name).parents if str(parent) != '.'}
    require(names == set(expected) and directories == required_dirs, 'exact artifact inventory')
    for name, item in expected.items():
        require(set(item) == {'sha256', 'bytes'} and type(item['bytes']) is int and item['bytes'] >= 0, 'artifact identity schema')
        path = child(root, name)
        require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], 'artifact changed')
