#!/usr/bin/env python3
"""Run operator-configured real-weight evidence jobs, never edit capability citations.

The config contains paths to an approved catalog and conformance registration, not
credentials. It is read from the trusted runner, never supplied by a pull request.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--native', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not isinstance(config, dict) or set(config) - {'catalog', 'registration', 'benchmark_registration', 'generation_cases'}:
        parser.error('config permits only catalog, registration, benchmark_registration, generation_cases')
    paths = {}
    for key in ('catalog', 'registration'):
        paths[key] = Path(config[key]).resolve(strict=True)
        if not paths[key].is_file():
            parser.error(f'{key} must name a file')
    native = args.native.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    import torch
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        parser.error('this evidence job requires the reference RTX 4090 on cuda:0')
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    expected = {}
    for extra in ('inference', 'inference-audio', 'inference-optimize', 'inference-video'):
        for requirement in project['optional-dependencies'].get(extra, []):
            package, version = requirement.split('==', 1)
            expected[package] = version
    installed = {name: importlib.metadata.version(name) for name in expected}
    for name, version in expected.items():
        if installed[name].split('+', 1)[0] != version:
            parser.error(f'{name} must match repository pin {version}; found {installed[name]}')
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'device': torch.cuda.get_device_name(0),
        'total_memory_mb': torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
        'python': sys.version.split()[0],
        'packages': installed,
        'results': [],
        'qualification': 'Observed conformance and benchmark evidence only; image/audio quality and benchmark citation require maintainer review.',
    }
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}

    def run(label, command, timeout):
        started = time.monotonic()
        log = output / f'{label}.log'
        with log.open('w') as stream:
            try:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
                stream.write('\nTimed out; no successful evidence claimed.\n')
        report['results'].append({'check': label, 'status': 'pass' if code == 0 else 'fail', 'exit_code': code, 'seconds': round(time.monotonic() - started, 3), 'log': log.name})
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    with tempfile.TemporaryDirectory(prefix='media-gpu-') as temporary:
        for backend in ('diffusers-sdxl', 'owned-sdxl', 'native-sdxl'):
            launch = [str(native)] if backend == 'native-sdxl' else [sys.executable, '-m', 'media_engine.server', '--backend', backend]
            launch += ['--catalog', str(paths['catalog']), '--directory', str(Path(temporary) / backend), '--device', 'cuda:0', '--port', '0', '--max-retained', '10']
            run('conformance-' + backend, [sys.executable, '-m', 'tests.conformance.run', '--registration', str(paths['registration']), '--verbose', '--launch', *launch], 3600)
        run('offload-reference-families', [sys.executable, 'packaging/offload-evidence.py', '--catalog', str(paths['catalog']), '--registration', str(paths['registration']), '--output', str(output)], 10800)
        run('generation-features', [sys.executable, 'packaging/generation-evidence.py', '--catalog', str(paths['catalog']), '--cases', config.get('generation_cases', ''), '--output', str(output / 'generation')], 10800)
        if os.environ.get('RUN_BENCHMARK') == 'true':
            registration = Path(config['benchmark_registration']).resolve(strict=True)
            for backend in ('diffusers-sdxl', 'owned-sdxl'):
                # Repeat once with identical settings; retain both artifacts for digest comparison.
                for repetition in (1, 2):
                    run(f'benchmark-{backend}-{repetition}', [sys.executable, 'tools/benchmark.py', '--backend', backend, '--device', 'cuda:0', '--catalog', str(paths['catalog']), '--registration', str(registration), '--precision', 'float16', '--width', '1024', '--height', '1024', '--steps', '30', '--runs', '5', '--option', 'precision=bfloat16', '--option', 'attention=flash', '--option', 'quantization=int8-weight-only', '--option', 'compile=inductor', '--output', str(output), '--label', f'{backend}-{repetition}'], 3600)
    return int(any(result['status'] == 'fail' for result in report['results']))


if __name__ == '__main__':
    raise SystemExit(main())
