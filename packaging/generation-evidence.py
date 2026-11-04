#!/usr/bin/env python3
"""Execute operator-supplied real-weight protocol 1.4 cases and record evidence.

The case file is local to the trusted runner, never a PR input. Missing coverage
fails the job; successful tiny fixtures are not accepted as real-weight evidence.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from media_engine.backend import DiffusersBackend
from media_engine.contracts import PROTOCOL
from media_engine.contracts import WorkerError
from media_engine.server import Worker
from media_engine import mp4, png

REQUIRED = {
    *(f'{family}:{task}' for family in ('flux-schnell', 'sd35')
      for task in ('text-to-image', 'image-to-image', 'inpaint')),
    'sdxl-base:inpaint', 'controlnet', 'textual-inversion', 'ip-adapter',
    'vae', 'clip-skip', 'previews', 'mp4', 'lora',
    *(f'image-transform:{op}' for op in ('resize', 'crop', 'pad', 'composite', 'canny')),
}


def wait(worker, operation=None, execution=None):
    deadline = time.monotonic() + 1800
    fetched = {}
    while time.monotonic() < deadline:
        snap = worker.operations[operation] if operation else worker.snapshot(execution)
        if execution and snap['state'] == 'running':
            for event in snap['events']:
                if event['type'] == 'preview' and event['data']['preview_id'] not in fetched:
                    identifier = event['data']['preview_id']
                    try:
                        data = worker.preview(identifier)
                        width, height, _ = png.read_rgb(data)
                        fetched[identifier] = {'width': width, 'height': height, 'sha256': hashlib.sha256(data).hexdigest()}
                    except WorkerError as error:
                        if error.status != 410:
                            raise
        if snap['state'] != 'running':
            if snap['state'] != 'succeeded':
                raise RuntimeError('worker operation failed: ' + snap.get('error', {}).get('code', 'unknown'))
            return {**snap, 'fetched_previews': list(fetched.values())}
        time.sleep(.02)
    if execution:
        worker.cancel(execution)
    raise TimeoutError('worker operation exceeded 30 minutes')


def coverage(load, request, snapshots):
    tags = {load['bundle']['family'] + ':' + request['task']}
    if load.get('controlnets'):
        tags.add('controlnet')
    if load.get('adapters'):
        tags.add('lora')
    tags.update(use['component']['kind'] for use in load.get('conditioning', []))
    if request['parameters'].get('clip_skip'):
        tags.add('clip-skip')
    if request['task'] == 'image-transform':
        tags.add('image-transform:' + request['parameters']['operation'])
    if all(snap.get('fetched_previews') for snap in snapshots):
        tags.add('previews')
    if all(any(out['mime_type'] == 'video/mp4' for out in snap['outputs']) for snap in snapshots):
        tags.add('mp4')
    return tags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    config = json.loads(args.cases.read_text())
    if config.get('schema_version') != 1 or not config.get('cases'):
        parser.error('case file needs schema_version 1 and a nonempty cases list')
    import torch
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        parser.error('reference RTX 4090 required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'cases': [], 'covered': [], 'missing': sorted(REQUIRED), 'quality_review': 'required; no automatic quality or optimization approval'}
    covered, names = set(), set()
    for case in config['cases']:
        name = case.get('name', '')
        if not re.fullmatch('[a-z0-9][a-z0-9-]{0,63}', name) or name in names:
            parser.error('case names must be unique simple lowercase identifiers')
        names.add(name)
        row = {'name': name, 'status': 'fail', 'runs': []}
        backend = None
        try:
            if case.get('weight_evidence') != 'operator-approved-real-weights':
                raise ValueError('explicit real-weight attestation required')
            load = json.loads(Path(case['load']).read_text())
            template = json.loads(Path(case['request']).read_text())
            row['load_sha256'] = hashlib.sha256(json.dumps(load, sort_keys=True).encode()).hexdigest()
            row['request_sha256'] = hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest()
            with tempfile.TemporaryDirectory(prefix='media-generation-') as state:
                backend = DiffusersBackend('cuda:0')
                worker = Worker(state, backend, catalog, max_storage_mb=1 << 20)
                load['protocol'] = PROTOCOL
                started = time.monotonic()
                loaded = wait(worker, operation=worker.control('load', load)['operation_id'])
                row['load_seconds'] = time.monotonic() - started
                row['memory_estimate'] = loaded.get('memory_estimate')
                inputs = {}
                for key, filename in case.get('input_files', {}).items():
                    inputs[key] = worker.store_input(Path(filename).read_bytes())['input_id']
                snapshots = []
                for repetition in (1, 2):
                    request = copy.deepcopy(template)
                    request.update(protocol=PROTOCOL, bundle_digest=load['bundle']['bundle_digest'], attempt_id=f'attempt_{name}_{repetition}')
                    request['inputs'].update(inputs)
                    started = time.monotonic()
                    snapshot = wait(worker, execution=worker.submit(request)['execution_id'])
                    snapshots.append(snapshot)
                    observed = []
                    for item in snapshot['outputs']:
                        path = worker.output_path(snapshot['execution_id'], item['output_id'])[0]
                        data = path.read_bytes()
                        checksum = hashlib.sha256(data).hexdigest()
                        if checksum != item['sha256']:
                            raise ValueError('output digest mismatch')
                        facts = mp4.read_metadata(path) if item['mime_type'] == 'video/mp4' else None
                        if item['mime_type'] == 'image/png':
                            png.read_rgb(data)
                        # Preserve generated media for the operator's visual quality review.
                        target = output / f'{name}-{repetition}-{item["output_id"]}{path.suffix}'
                        target.write_bytes(data)
                        observed.append({'output_id': item['output_id'], 'mime_type': item['mime_type'], 'sha256': checksum, 'file': target.name, 'video': facts})
                    row['runs'].append({'seconds': time.monotonic() - started, 'measurements': snapshot.get('measurements'), 'outputs': observed, 'previews': snapshot['fetched_previews']})
                    worker.release(snapshot['execution_id'])
                first, second = ([o['sha256'] for o in run['outputs']] for run in row['runs'])
                row['repeat_identical'] = first == second
                if first != second:
                    raise ValueError('repeated seeded output digests differ')
                tags = coverage(load, request, snapshots)
                row['coverage'] = sorted(tags)
                covered.update(tags)
                row['status'] = 'pass'
        except Exception as error:
            row['error_type'] = type(error).__name__
        finally:
            if backend is not None:
                backend.unload()
        report['cases'].append(row)
        report.update(covered=sorted(covered), missing=sorted(REQUIRED - covered))
        (output / 'generation-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return int(bool(report['missing']) or any(case['status'] != 'pass' for case in report['cases']))


if __name__ == '__main__':
    raise SystemExit(main())
