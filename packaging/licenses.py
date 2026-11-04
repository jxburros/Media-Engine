#!/usr/bin/env python3
"""Collect dependency license notices into a self-contained release asset."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--kind', choices=['go', 'rust'], required=True)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.kind == 'rust':
    command = ['cargo', 'metadata', '--locked', '--format-version', '1', '--manifest-path', 'native/Cargo.toml', '--filter-platform', 'x86_64-unknown-linux-gnu']
    if args.cuda:
        command += ['--features', 'cuda']
    metadata = json.loads(subprocess.check_output(command, text=True))
    nodes = {node['id']: node for node in metadata['resolve']['nodes']}
    pending, linked = [metadata['resolve']['root']], set()
    while pending:
        package_id = pending.pop()
        if package_id in linked:
            continue
        linked.add(package_id)
        pending.extend(dep['pkg'] for dep in nodes[package_id]['deps']
                       if any(kind['kind'] is None for kind in dep['dep_kinds']))
    packages = {(p['name'], p['version']): p for p in metadata['packages']
                if p['source'] and p['id'] in linked}
    dependencies = [(p['name'], p['version'], Path(p['manifest_path']).parent) for p in packages.values()]
else:
    # Inspect the binary's import graph, not unused/test-only modules in go.mod.
    # Every selected module has been downloaded by the preceding binary build.
    raw = subprocess.check_output(['go', 'list', '-deps', '-json', './cmd/media-runtime'], text=True)
    decoder = json.JSONDecoder()
    modules = {}
    while raw.strip():
        item, end = decoder.raw_decode(raw.lstrip())
        raw = raw.lstrip()[end:]
        module = item.get('Module', {})
        if module and not module.get('Main'):
            modules[module['Path']] = (module['Path'], module['Version'], Path(module['Dir']))
    dependencies = list(modules.values())
notices = []
for name, version, root in sorted(dependencies):
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name.upper().startswith(('LICENSE', 'LICENCE', 'COPYING', 'NOTICE')) and p.stat().st_size < 1024 * 1024)
    if not files and args.kind == 'rust':
        # A subcrate can omit a duplicate copy of its repository-wide license.
        # Reuse only a same-repository, same-license sibling's actual notice.
        package = packages[name, version]
        for sibling in packages.values():
            if sibling['repository'] and sibling['repository'].rstrip('/') == (package['repository'] or '').rstrip('/') and sibling['license'] == package['license']:
                sibling_root = Path(sibling['manifest_path']).parent
                candidates = sorted(p for p in sibling_root.glob('*') if p.is_file() and p.name.upper().startswith(('LICENSE', 'LICENCE', 'COPYING', 'NOTICE')))
                if candidates:
                    root, files = sibling_root, candidates
                    break
    if not files:
        if args.kind == 'rust' and name in ('ug', 'ug-cuda') and version == '0.5.0':
            package = packages[name, version]
            vcs = json.loads((root / '.cargo_vcs_info.json').read_text())
            if (package['repository'] == 'https://github.com/LaurentMazare/ug'
                    and package['license'] == 'MIT OR Apache-2.0'
                    and vcs['git']['sha1'] == '8c6dd50d6e96a22db70e1462c0e49d0cda8294f7'):
                root = Path(__file__).resolve().parent / 'notices' / 'ug-0.5.0'
                files = [root / 'LICENSE-MIT', root / 'LICENSE-APACHE']
    if not files:
        raise SystemExit(f'Missing license notice for {name} {version}; inspect the dependency before packaging')
    notices.append(f'=== {name} {version} ===\n')
    for path in files:
        notices.append(f'--- {path.relative_to(root)} ---\n{path.read_text(errors="replace")}\n')
args.output.write_text('\n'.join(notices))
