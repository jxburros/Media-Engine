"""Run the conformance suite and report per-section results.

    python -m tests.conformance.run --url http://127.0.0.1:7811 --registration reg.json
    python -m tests.conformance.run --registration ROOT/conformance.json \\
        --launch python tests/serve_fixture.py ROOT

Over network mode (an https:// URL, or a launched worker whose startup record says
`scheme: "https"`) pass --tls-ca, --tls-cert and --tls-key for mutual TLS.

With --url the credential is read from the environment variable named by --token-env. With
--launch the suite starts the worker itself (generating a credential, reading the startup record)
and stops it afterwards; only a launched worker can pass the startup-record check. A worker is
reported conformant only when every section passed and none was skipped.
"""

import argparse
import json
import os
import secrets
import subprocess
import sys
import unittest

from .suite import SECTIONS, SUITE_VERSION, load_suite


class SectionResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sections = {section.section: "pass" for section in SECTIONS}

    def mark(self, test, outcome):
        section = getattr(test, "section", None)
        if section is None:  # a class-level setUp error is reported against a placeholder
            name = str(test)
            section = next((s.section for s in SECTIONS if s.__name__ in name), "setup")
        current = self.sections.get(section, "pass")
        order = {"pass": 0, "skip": 1, "fail": 2}
        if order[outcome] > order[current]:
            self.sections[section] = outcome

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.mark(test, "fail")

    def addError(self, test, err):
        super().addError(test, err)
        self.mark(test, "fail")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.mark(test, "skip")


def run_suite(stream=sys.stderr, verbosity=1):
    """Run every section with the current environment; returns (conformant, {section: outcome})."""
    runner = unittest.TextTestRunner(stream=stream, verbosity=verbosity, resultclass=SectionResult)
    result = runner.run(load_suite())
    sections = result.sections
    conformant = result.testsRun > 0 and all(v == "pass" for v in sections.values())
    return conformant, sections


def launch(command, token):
    environment = {**os.environ, "MEDIA_ENGINE_TOKEN": token}
    process = subprocess.Popen(command, stdout=subprocess.PIPE, env=environment, text=True)
    line = process.stdout.readline()
    if not line:
        process.kill()
        raise SystemExit("the launched worker exited before printing its startup record")
    return process, json.loads(line)


def launched_url(record):
    """The URL of a worker the suite launched, from its startup record."""
    host = record.get("bind") or "127.0.0.1"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"{record.get('scheme', 'http')}://{host}:{record['port']}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="media-worker protocol conformance suite")
    parser.add_argument("--url")
    parser.add_argument("--token-env", default="MEDIA_ENGINE_TOKEN")
    parser.add_argument("--registration", help="JSON {bundle, upscaler?, video?, options?}")
    parser.add_argument("--startup-record", help="file holding the worker's startup record")
    parser.add_argument("--launch", nargs=argparse.REMAINDER, help="command that starts the worker")
    parser.add_argument("--tls-ca", help="PEM CA that issued the worker's certificate")
    parser.add_argument("--tls-cert", help="PEM client certificate for mutual TLS")
    parser.add_argument("--tls-key", help="private key of --tls-cert")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    for name, value in (("CA", args.tls_ca), ("CERT", args.tls_cert), ("KEY", args.tls_key)):
        if value:
            os.environ["MEDIA_ENGINE_TLS_" + name] = os.path.abspath(value)
    process = None
    if args.launch:
        token = secrets.token_hex(32)
        process, record = launch(args.launch, token)
        os.environ.update(
            MEDIA_ENGINE_URL=launched_url(record),
            MEDIA_ENGINE_TOKEN=token,
            MEDIA_ENGINE_STARTUP_RECORD=json.dumps(record),
        )
    elif args.url:
        os.environ["MEDIA_ENGINE_URL"] = args.url
        os.environ["MEDIA_ENGINE_TOKEN"] = os.environ.get(args.token_env, "")
        if args.startup_record:
            os.environ["MEDIA_ENGINE_STARTUP_RECORD"] = open(args.startup_record).read()
    else:
        parser.error("--url or --launch is required")
    if args.registration:
        os.environ["MEDIA_ENGINE_REGISTRATION"] = os.path.abspath(args.registration)
    try:
        conformant, sections = run_suite(verbosity=2 if args.verbose else 1)
    finally:
        if process:
            process.terminate()
            process.wait(10)
    for section, outcome in sections.items():
        print(f"{outcome.upper():4} {section}")
    if conformant:
        print(f"conformant: {SUITE_VERSION}")
        return 0
    skipped = [s for s, o in sections.items() if o == "skip"]
    print(f"not conformant ({SUITE_VERSION})" + (f": skipped sections {', '.join(skipped)}" if skipped else ""))
    return 1


if __name__ == "__main__":
    sys.exit(main())
