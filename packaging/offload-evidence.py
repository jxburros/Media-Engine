#!/usr/bin/env python3
"""Measure explicit none/model/sequential offload on approved CUDA reference bundles.

Records actual load/execute times, device peaks, output digests and differences.
Never updates the optimization index or chooses a mode for callers.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from benchmark import benchmark, request_set
from media_engine import png
from media_engine.backend import backend_class
from media_engine.server import Worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    registrations = json.loads(args.registration.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    cases = (
        ('sdxl-base', 'bundle', (1024, 1024), 30, 10),
        ('svd', 'video', (1024, 576), 25, 10),
        ('stable-audio-open', 'audio', (1024, 1024), 100, 10),
        ('stable-audio-open', 'audio', (1024, 1024), 100, 47),
    )
    failed = False
    for family, key, size, steps, duration in cases:
        label = f'offload-{family}' + (f'-{duration}s' if family == 'stable-audio-open' else '')
        try:
            registration = registrations[key]
            if registration['family'] != family:
                raise ValueError(f'{key} registration must have family {family}')
            with tempfile.TemporaryDirectory(prefix='media-offload-') as state:
                backend = backend_class('diffusers-sdxl')('cuda:0')
                worker = Worker(state, backend, catalog, max_storage_mb=1 << 20)
                try:
                    options = [{**backend.capabilities()['load_defaults'], 'precision': 'float16', 'offload': mode} for mode in ('none', 'model', 'sequential')]
                    image = png.write(*size, bytes((128, 96, 64)) * (size[0] * size[1])) if family == 'svd' else None
                    document = benchmark(worker, registration, family, options, 2,
                                         request_set(family, size, steps, duration), image,
                                         note='Explicit offload comparison on reference hardware; no mode is selected automatically.')
                finally:
                    backend.unload()
            document['tool'] = 'packaging/offload-evidence.py'
            document.pop('thresholds', None)
            for result in document['results']:
                result.get('summary', {}).pop('qualifies', None)
                if 'error' in result:
                    failed = True
            (args.output / f'{label}.json').write_text(json.dumps(document, indent=2) + '\n')
        except Exception as error:
            failed = True
            # Keep a failed case visible while continuing the independent families.
            (args.output / f'{label}.json').write_text(json.dumps({'family': family, 'status': 'fail', 'error_type': type(error).__name__}) + '\n')
            print(f'{label}: failed ({type(error).__name__})', file=sys.stderr)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
