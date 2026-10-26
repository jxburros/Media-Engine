"""Launch the real HTTP worker with an explicitly identified test backend."""

import json
import os
import sys
from pathlib import Path
from fixture import TestBackend, make_adapter, make_bundle
from media_engine.server import Worker, create_server

root = Path(sys.argv[1]).resolve()
make_bundle(root / "model")
make_adapter(root / "lora")
worker = Worker(
    root / "work",
    TestBackend(),
    {"fixture": str(root / "model"), "fixture-lora": str(root / "lora")},
)
server = create_server(worker, os.environ["MEDIA_ENGINE_TOKEN"], 0)
print(
    json.dumps(
        {
            "protocol": "media-worker/1.1",
            "supported_protocols": ["media-worker/1.0", "media-worker/1.1"],
            "worker_version": "0.3.0-test-fixture",
            "backend_id": worker.backend.backend_id,
            "session_id": worker.session_id,
            "pid": os.getpid(),
            "port": server.server_port,
            "test_fixture": True,
        }
    ),
    flush=True,
)
server.serve_forever()
