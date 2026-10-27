"""Launch the real HTTP worker with an explicitly identified test backend.

Usage: python tests/serve_fixture.py ROOT [options]   (MEDIA_ENGINE_TOKEN must be set)

The first stdout line is the startup record. ROOT receives these catalog handles:
  fixture      sdxl-base bundle (zero-byte stand-in weights)        ROOT/model
  fixture-lora PEFT LoRA adapter                                     ROOT/lora
  upscaler     esrgan single-file-safetensors bundle, scale 2        ROOT/upscaler
  converted    sdxl-base bundle with conversion.json provenance      ROOT/converted
  checkpoint   bare single-file checkpoint (needs conversion)        ROOT/checkpoint
plus ROOT/catalog.json (handle → directory), ROOT/registrations.json (handle → bundle or
adapter registration) and ROOT/conformance.json (the conformance suite's registration document).
Every option also reads an environment variable so a supervisor's test can set it without
changing the launch command.

Network mode takes the worker's own flags and rules (media_engine.server.network_configuration):
--bind, --tls-cert, --tls-key, --tls-client-ca and --no-parent-watch, also read from
MEDIA_ENGINE_FIXTURE_BIND, _TLS_CERT, _TLS_KEY, _TLS_CLIENT_CA and _NO_PARENT_WATCH=1.
tests/certs holds a fixture CA, server and client certificates; copy server.key to a 0600 file
first, because the worker refuses a key readable by group or others.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from fixture import (
    FIXTURE_VERSION,
    TestBackend,
    make_adapter,
    make_bundle,
    make_checkpoint,
    make_converted,
    make_esrgan_bundle,
)
from media_engine.contracts import SUPPORTED_PROTOCOLS
from media_engine.server import (
    NetworkError,
    Worker,
    add_network_arguments,
    create_server,
    launch_protocol,
    network_configuration,
)


def env(name, default=None):
    return os.environ.get("MEDIA_ENGINE_FIXTURE_" + name, default)


def number(name):
    value = env(name)
    return int(value) if value not in (None, "") else None


parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("root")
parser.add_argument(
    "--protocol",
    default=env("PROTOCOL", SUPPORTED_PROTOCOLS[-1]),
    choices=SUPPORTED_PROTOCOLS[1:],
    help="highest minor to serve (media-worker/1.1 reproduces the 0.3.0 fixture surface)",
)
parser.add_argument(
    "--device-unavailable",
    action="store_true",
    default=env("DEVICE_UNAVAILABLE") == "1",
    help="report the selected device as unavailable (exercises the authorized cpu fallback)",
)
parser.add_argument("--device-index", type=int, default=number("DEVICE_INDEX") or 0)
parser.add_argument("--host-id", default=env("HOST_ID", "fixture-host"))
parser.add_argument(
    "--precisions",
    default=env("PRECISIONS", "float32"),
    help="comma-separated advertised precisions, e.g. float16,float32 to emulate a GPU",
)
parser.add_argument("--memory-budget-mb", type=int, default=number("MEMORY_BUDGET_MB"))
parser.add_argument(
    "--estimate-mb",
    type=int,
    default=number("ESTIMATE_MB"),
    help="override the float32 memory estimate (float16 is half) to drive budget fallbacks",
)
parser.add_argument("--output-side", type=int, default=number("OUTPUT_SIDE"))
parser.add_argument("--max-retained", type=int, default=number("MAX_RETAINED") or 128)
parser.add_argument("--max-storage-mb", type=int, default=number("MAX_STORAGE_MB") or 1024)
add_network_arguments(parser)
parser.set_defaults(
    bind=env("BIND", "127.0.0.1"),
    tls_cert=env("TLS_CERT"),
    tls_key=env("TLS_KEY"),
    tls_client_ca=env("TLS_CLIENT_CA"),
    no_parent_watch=env("NO_PARENT_WATCH") == "1",
)
args = parser.parse_args()
try:
    bind, tls = network_configuration(args)
except NetworkError as exc:
    parser.error(str(exc))
protocols = SUPPORTED_PROTOCOLS[: SUPPORTED_PROTOCOLS.index(args.protocol) + 1]
precisions = [p for p in args.precisions.split(",") if p]
if not precisions or any(p not in ("float16", "float32") for p in precisions):
    parser.error("precisions must be float16 and/or float32")

root = Path(args.root).resolve()
handles = {
    "fixture": root / "model",
    "fixture-lora": root / "lora",
    "upscaler": root / "upscaler",
    "converted": root / "converted",
    "checkpoint": root / "checkpoint",
}
registrations = {
    "fixture": make_bundle(handles["fixture"]),
    "fixture-lora": make_adapter(handles["fixture-lora"]),
    "upscaler": make_esrgan_bundle(handles["upscaler"]),
}
registrations["converted"] = make_converted(handles["converted"], reference=handles["fixture"])
make_checkpoint(handles["checkpoint"])
catalog = {handle: str(path) for handle, path in handles.items()}
(root / "catalog.json").write_text(json.dumps(catalog, indent=2))
(root / "registrations.json").write_text(json.dumps(registrations, indent=2))
(root / "conformance.json").write_text(
    json.dumps(
        {
            "bundle": registrations["fixture"],
            "upscaler": registrations["upscaler"],
            "adapter": registrations["fixture-lora"],
        },
        indent=2,
    )
)
backend = TestBackend(
    protocols=protocols,
    device_unavailable=args.device_unavailable,
    precisions=sorted(precisions, key=lambda p: p != "float16") if "float16" in precisions else precisions,
    memory_budget_mb=args.memory_budget_mb,
    estimate_mb=args.estimate_mb,
    output_side=args.output_side,
    device_index=args.device_index,
    host_id=args.host_id,
)
worker = Worker(
    root / "work",
    backend,
    catalog,
    max_retained=args.max_retained,
    max_storage_mb=args.max_storage_mb,
    protocols=protocols,
)
server = create_server(
    worker,
    os.environ["MEDIA_ENGINE_TOKEN"],
    0,
    bind,
    tls,
    ownership_available=not args.no_parent_watch,
)
record = {
    "protocol": launch_protocol(protocols),  # see media_engine.server.startup_record
    "supported_protocols": list(protocols),
    "highest_protocol": protocols[-1],
    "worker_version": FIXTURE_VERSION,
    "backend_id": worker.backend.backend_id,
    "session_id": worker.session_id,
    "pid": os.getpid(),
    "port": server.server_port,
    "test_fixture": True,
}
if worker.speaks_1_2:
    record["device_identity"] = worker.device_identity()
    record.update(worker.transport)
print(json.dumps(record), flush=True)
sys.stdout.flush()
server.serve_forever()
