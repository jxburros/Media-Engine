#!/usr/bin/env python3
"""Fail packaging on unexpected or unresolved shared libraries (Linux x64)."""
import argparse
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('binary')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()
result = subprocess.run(['ldd', args.binary], text=True, capture_output=True)
output = result.stdout + result.stderr
if 'not a dynamic executable' in output or 'statically linked' in output:
    raise SystemExit(0)
if result.returncode:
    raise SystemExit('ldd failed: ' + output)
allowed = re.compile(r'^(?:linux-vdso\.so\.1|ld-linux-x86-64\.so\.2|lib(?:c|m|dl|pthread|rt|gcc_s)\.so\.[0-9]+)$')
cuda = re.compile(r'^lib(?:cuda|cublas|cublasLt|curand|nvrtc|nvrtc-builtins|nvJitLink|stdc\+\+)\.so(?:\.[0-9]+)+$')
for line in output.splitlines():
    name = line.strip().split()[0].rsplit('/', 1)[-1]
    if not (allowed.fullmatch(name) or (args.cuda and cuda.fullmatch(name))):
        raise SystemExit('Unexpected release dependency: ' + name)
    # libcuda is supplied by the host driver, absent on compile-only CI hosts.
    if 'not found' in line and not (args.cuda and name == 'libcuda.so.1'):
        raise SystemExit('Unresolved release dependency: ' + name)
print(output, end='')
