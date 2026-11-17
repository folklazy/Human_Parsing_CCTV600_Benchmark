"""Small durable records and workspace-relative source bindings."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def event(path, **value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(dict(at_utc=now(), **value), allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, path)


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def install_shared_paths():
    import sys
    for relative in ('Human_Parsing_Benchmark/src', 'CCTV_Human_Parsing_Benchmark/scripts',
                     'YOLO_Instance_Segmentation_CCTV600_Scaling_Study/src'):
        sys.path.insert(0, str(WORKSPACE / relative))


def verify_sources(config):
    for relative, digest in config['source_hashes'].items():
        if sha(WORKSPACE / relative) != digest:
            raise ValueError('Frozen source changed: ' + relative)


def membership(expected, actual):
    expected, actual = list(expected), list(actual)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    duplicate = len(actual) - len(set(actual))
    return dict(expected=len(expected), completed=len(set(actual)), missing=len(missing),
                unexpected=len(unexpected), duplicate=duplicate,
                strict_complete=not (missing or unexpected or duplicate), missing_ids=missing)
