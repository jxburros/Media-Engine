"""media-worker/1.2: authenticated loopback, one warm pipeline, no waiting queue."""

import argparse
import copy
import hashlib
import hmac
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
from . import png
from .backend import Cancelled, backend_class
from .contracts import (
    FAMILIES,
    LIMITS,
    PROTOCOL,
    SUPPORTED_PROTOCOLS,
    WORKER_VERSION,
    WorkerError,
    effective_settings,
    effective_steps,
    family_facts,
    minor,
    output_ids,
    png_dimensions,
    strict,
    protocol,
    bundle_path,
    validate_adapters,
    validate_parameters,
)

TERMINAL = {"succeeded", "failed", "cancelled"}
PRIVATE = {"cancel", "request", "image_path", "expected_outputs"}
OUTPUT_ID = r"[a-z][a-z0-9_]{0,31}"


def error(code, field=None):
    result = {"code": code, "message": code.replace("_", " "), "retryable": False}
    if field:
        result["field"] = field
    if code == "out_of_memory":
        result["restart_required"] = True
    return result


class MissingOutput(Exception):
    def __init__(self, output_id):
        self.output_id = output_id


class Worker:
    def __init__(
        self,
        directory,
        backend,
        catalog,
        max_retained=128,
        max_storage_mb=1024,
        max_input_mb=32,
        protocols=SUPPORTED_PROTOCOLS,
    ):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.directory / "inputs").mkdir(exist_ok=True, mode=0o700)
        self.backend, self.catalog = backend, catalog
        self.protocols = tuple(protocols)
        self.session_id = str(uuid.uuid4())
        self.lock = threading.RLock()
        self.executions, self.attempts, self.operations, self.inputs = {}, {}, {}, {}
        self.active = None
        self.state = "ready"
        self.loaded = None
        self.loaded_family = None
        self.loaded_facts = {}
        self.max_retained, self.max_storage_mb = max_retained, max_storage_mb
        self.max_input_mb = max_input_mb

    @property
    def speaks_1_2(self):
        return "media-worker/1.2" in self.protocols

    def check(self, body):
        """A worker pinned to an older minor refuses newer minors exactly as a 0.3.0 worker would."""
        if isinstance(body, dict):
            version = body.get("protocol")
            if version in SUPPORTED_PROTOCOLS and version not in self.protocols:
                raise WorkerError(400, "protocol_mismatch", "protocol")

    def capabilities(self):
        backend = self.backend.capabilities()
        return {
            **backend,
            "session_id": self.session_id,
            "formats": backend.get("formats", ["diffusers-safetensors"]),
            "output_formats": ["png"],
            "input_formats": ["image/png"],
            "progress": True,
            "cancellation": True,
            "retention": "until-discard-or-session-end",
            "max_retained_executions": self.max_retained,
            "max_storage_mb": self.max_storage_mb,
            "max_input_mb": self.max_input_mb,
            "required_roles": [
                "unet",
                "vae",
                "text_encoder",
                "text_encoder_2",
                "tokenizer",
                "tokenizer_2",
                "scheduler",
            ],
        }

    def device_identity(self):
        return self.backend.device_identity() if self.speaks_1_2 else None

    def info(self):
        caps = self.capabilities()
        with self.lock:
            info = {
                "protocol": self.protocols[-1],
                "supported_protocols": list(self.protocols),
                "worker_version": WORKER_VERSION,
                "backend_id": self.backend.backend_id,
                "ownership": self.backend.ownership,
                "session_id": self.session_id,
                "state": self.state,
                "service_ready": self.state != "error",
                "device_ready": caps["ready"],
                "model_ready": self.loaded is not None,
                "loaded_bundle": self.loaded,
                "device": caps["device"],
                "build": {"package": "media-engine", "version": WORKER_VERSION},
            }
        if self.speaks_1_2:
            info["device_identity"] = self.device_identity()
        return info

    def validate_load(self, body, verify=True):
        self.check(body)
        strict(body, ("protocol", "bundle", "options"), ("adapters", "authorized_fallbacks"))
        version = protocol(body)
        level = minor(version)
        if "authorized_fallbacks" in body and level < 2:
            raise WorkerError(400, "unsupported", "authorized_fallbacks")
        strict(body["options"], ("precision", "attention", "offload"))
        options = body["options"]
        caps = self.capabilities()
        precisions = list(caps["precisions"])
        if level >= 2 and "precision" in caps.get("auto_options", []):
            precisions.append("auto")
        if (
            options["precision"] not in precisions
            or options["attention"] != "sdpa"
            or options["offload"] != "none"
        ):
            raise WorkerError(400, "unsupported", "options")
        authorized = body.get("authorized_fallbacks", [])
        if (
            not isinstance(authorized, list)
            or any(not isinstance(f, str) or f not in caps.get("fallbacks", []) for f in authorized)
            or len(set(authorized)) != len(authorized)
        ):
            raise WorkerError(400, "invalid_request", "authorized_fallbacks")
        root = bundle_path(body["bundle"], self.catalog, verify, version)
        family = body["bundle"]["family"]
        adapters = validate_adapters(body.get("adapters", []), self.catalog, version, verify)
        if adapters and family != "sdxl-base":
            raise WorkerError(400, "unsupported", "adapters")
        plan = self.backend.plan(root, options, adapters, family, authorized)
        return root, adapters, family, plan

    def estimate(self, root, plan, adapters, family, level):
        estimate = self.backend.estimate(
            root, plan["options"], adapters, family, plan["device"]
        )
        if level >= 2 and family == "sdxl-base":
            estimate = {
                **estimate,
                "count": {
                    "limit": LIMITS["count"],
                    "scaling": "activation working memory grows linearly with count; activations are excluded from value_mb",
                },
            }
        return estimate

    def validation(self, body):
        """The /v1/validate response for a load body."""
        root, adapters, family, plan = self.validate_load(body)
        level = minor(body["protocol"])
        result = {
            "valid": True,
            "memory_estimate": self.estimate(root, plan, adapters, family, level),
            "adapters": [
                {"adapter_id": r["adapter_id"], "layout": layout}
                for r, _a, _w, _s, layout in adapters
            ],
        }
        if level >= 2:
            result.update(
                family=family,
                tasks=list(FAMILIES[family]["tasks"]),
                resolved_options=plan["options"],
                resolutions=plan["resolutions"],
                applied_fallbacks=plan["applied_fallbacks"],
                **family_facts(root, family),
            )
        return result

    def control(self, action, body):
        if action == "load":
            _root, _adapters, _family, plan = self.validate_load(body, verify=False)
            caps = self.capabilities()
            if caps.get("missing_dependencies") or not (
                caps["device_available"] or plan["device"] == "cpu"
            ):
                # An unavailable device is refused unless the caller authorized the CPU fallback.
                raise WorkerError(503, "backend_unavailable")
        else:
            self.check(body)
            strict(body, ("protocol",))
            protocol(body)
        with self.lock:
            if self.active or self.state in {"loading", "unloading", "error"}:
                raise WorkerError(409, "busy")
            if len(self.operations) >= self.max_retained:
                # Control records are also retained, never silently evicted.
                raise WorkerError(503, "storage_full")
            operation = {
                "operation_id": "op_" + uuid.uuid4().hex,
                "state": "running",
                "action": action,
                "error": None,
            }
            self.operations[operation["operation_id"]] = operation
            self.state = "loading" if action == "load" else "unloading"

            def run():
                try:
                    measurements = self.backend.unload()
                    with self.lock:
                        operation["measurements"] = measurements
                        self.loaded, self.loaded_family, self.loaded_facts = None, None, {}
                    if action == "load":
                        level = minor(body["protocol"])
                        root, adapters, family, plan = self.validate_load(body)
                        estimate = self.estimate(root, plan, adapters, family, level)
                        self.backend.load(root, plan["options"], adapters, family, plan["device"])
                        self.backend.applied_fallbacks = plan["applied_fallbacks"]
                        facts = family_facts(root, family)
                        with self.lock:
                            self.loaded = {
                                "bundle_digest": body["bundle"]["bundle_digest"],
                                "bundle": body["bundle"],
                                "options": body["options"],
                                "adapters": [
                                    {
                                        "adapter_id": record["adapter_id"],
                                        "digest": record["digest"],
                                        "scale": scale,
                                        "layout": layout,
                                    }
                                    for record, _r, _w, scale, layout in adapters
                                ],
                                "memory_estimate": estimate,
                            }
                            if self.speaks_1_2:
                                self.loaded.update(
                                    family=family,
                                    tasks=list(FAMILIES[family]["tasks"]),
                                    resolved_options=plan["options"],
                                    applied_fallbacks=plan["applied_fallbacks"],
                                    **facts,
                                )
                            self.loaded_family, self.loaded_facts = family, facts
                            operation["memory_estimate"] = estimate
                            if level >= 2:
                                operation.update(
                                    resolved_options=plan["options"],
                                    resolutions=plan["resolutions"],
                                    applied_fallbacks=plan["applied_fallbacks"],
                                )
                    with self.lock:
                        operation["state"] = "succeeded"
                        self.state = "ready"
                except Exception as exc:
                    usable = True
                    try:
                        self.backend.unload()
                    except Exception:
                        usable = False
                    with self.lock:
                        self.loaded, self.loaded_family, self.loaded_facts = None, None, {}
                        operation["state"] = "failed"
                        operation["error"] = (
                            error(exc.code, exc.field)
                            if isinstance(exc, WorkerError)
                            else error("model_load_failed")
                        )
                        self.state = "ready" if usable else "error"
                        if not usable:
                            operation["error"]["restart_required"] = True

            threading.Thread(target=run, daemon=True).start()
            return copy.deepcopy(operation)

    def emit(self, execution_id, kind, data):
        with self.lock:
            entry = self.executions[execution_id]
            event = {
                "session_id": self.session_id,
                "execution_id": execution_id,
                "sequence": len(entry["events"]) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": kind,
                "data": data,
            }
            entry["events"].append(event)
            entry["last_sequence"] = event["sequence"]
            if kind == "stage":
                entry["stage"] = data["stage"]
            if kind == "progress":
                entry["progress"] = data

    def snapshot(self, execution_id):
        with self.lock:
            if execution_id not in self.executions:
                raise WorkerError(404, "execution_not_found")
            return copy.deepcopy(
                {
                    k: v
                    for k, v in self.executions[execution_id].items()
                    if k not in PRIVATE
                }
            )

    def storage_used(self):
        return sum(
            p.stat().st_size
            for pattern in ("exec_*/*.png", "inputs/*.png")
            for p in self.directory.glob(pattern)
        )

    def store_input(self, data):
        width, height = png_dimensions(data)
        with self.lock:
            if self.state == "error":
                raise WorkerError(409, "busy")
            if (
                len(self.inputs) + len(self.executions) >= self.max_retained
                or self.storage_used() + len(data) > self.max_storage_mb * 1024**2
            ):
                raise WorkerError(503, "storage_full")
            input_id = "input_" + uuid.uuid4().hex
            path = self.directory / "inputs" / (input_id + ".png")
            temporary = path.with_suffix(".part")
            with temporary.open("wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            record = {
                "input_id": input_id,
                "session_id": self.session_id,
                "mime_type": "image/png",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "width": width,
                "height": height,
            }
            self.inputs[input_id] = record
            return dict(record)

    def input_record(self, input_id):
        with self.lock:
            record = self.inputs.get(input_id)
            if not record:
                raise WorkerError(404, "input_not_found")
            return dict(record)

    def release_input(self, input_id):
        with self.lock:
            self.input_record(input_id)
            if any(
                e["request"]["inputs"].get("image") == input_id
                and e["state"] not in TERMINAL
                for e in self.executions.values()
            ):
                raise WorkerError(409, "busy")
            (self.directory / "inputs" / (input_id + ".png")).unlink(missing_ok=True)
            del self.inputs[input_id]

    def loaded_checks(self, body, task):
        """Checks that need the loaded bundle; they apply when the request names that bundle."""
        if not self.loaded or self.loaded["bundle_digest"] != body["bundle_digest"]:
            return {}
        if task not in FAMILIES[self.loaded_family]["tasks"]:
            raise WorkerError(400, "unsupported", "task")
        if task != "upscale":
            return {}
        scale = self.loaded_facts["scale"]
        params = body["parameters"]
        if max(params["width"], params["height"]) * scale > self.capabilities()["limits"]["output_side"]:
            # Decided before any computation; the worker never shrinks a result to fit.
            raise WorkerError(400, "invalid_request", "output_side")
        return {
            "scale": scale,
            "output_width": params["width"] * scale,
            "output_height": params["height"] * scale,
        }

    def validate_execution(self, body):
        """The /v1/validate response for an execution body."""
        self.check(body)
        task = validate_parameters(body)
        level = minor(body["protocol"])
        with self.lock:
            facts = self.loaded_checks(body, task)
        result = {"valid": True, "task": task}
        if task != "upscale":
            result["effective_steps"] = effective_steps(body["parameters"], task)
        if level >= 2:
            result["outputs"] = output_ids(task, body["parameters"])
            result["effective_settings"] = {
                **effective_settings(body["parameters"], task, body["protocol"]),
                **facts,
            }
        return result

    def submit(self, body):
        self.check(body)
        task = validate_parameters(body)
        attempt = body["attempt_id"]
        level = minor(body["protocol"])
        with self.lock:
            if attempt in self.attempts:
                entry = self.executions[self.attempts[attempt]]
                if entry["request"] != body:
                    raise WorkerError(409, "attempt_conflict")
                return self.snapshot(entry["execution_id"])
            facts = self.loaded_checks(body, task)
            if self.active or self.state != "ready":
                raise WorkerError(409, "busy")
            if not self.loaded or self.loaded["bundle_digest"] != body["bundle_digest"]:
                raise WorkerError(409, "model_not_ready")
            image_path = None
            if task in ("image-to-image", "upscale"):
                record = self.inputs.get(body["inputs"]["image"])
                if not record:
                    raise WorkerError(404, "input_not_found")
                if (record["width"], record["height"]) != (
                    body["parameters"]["width"],
                    body["parameters"]["height"],
                ):
                    # The request must state the image's real size; nothing is resized silently.
                    raise WorkerError(400, "invalid_request", "width")
                image_path = self.directory / "inputs" / (record["input_id"] + ".png")
            expected = output_ids(task, body["parameters"])
            width = facts.get("output_width", body["parameters"]["width"])
            height = facts.get("output_height", body["parameters"]["height"])
            reserve = (width * height * 4 + 65536) * len(expected)
            if (
                len(self.executions) + len(self.inputs) >= self.max_retained
                or self.storage_used() + reserve > self.max_storage_mb * 1024**2
            ):
                raise WorkerError(503, "storage_full")
            execution_id = "exec_" + uuid.uuid4().hex
            self.active, self.state = execution_id, "busy"
            self.attempts[attempt] = execution_id
            entry = {
                "attempt_id": attempt,
                "execution_id": execution_id,
                "session_id": self.session_id,
                "protocol": body["protocol"],
                "task": task,
                "state": "running",
                "stage": "upscaling" if task == "upscale" else "encoding",
                "progress": None,
                "events": [],
                "last_sequence": 0,
                "effective_settings": {
                    **effective_settings(body["parameters"], task, body["protocol"]),
                    **facts,
                },
                "bundle_digest": body["bundle_digest"],
                "implementation": self.capabilities(),
                "cancel_requested": False,
                "outputs": [],
                "error": None,
                "cancel": threading.Event(),
                "request": body,
                "image_path": image_path,
                "expected_outputs": expected,
            }
            fallbacks = getattr(self.backend, "applied_fallbacks", [])
            if level >= 2 or fallbacks:
                entry["applied_fallbacks"] = copy.deepcopy(fallbacks)
            self.executions[execution_id] = entry
            self.emit(execution_id, "state", {"state": "running"})
            threading.Thread(target=self.run, args=(execution_id,), daemon=True).start()
            return self.snapshot(execution_id)

    def finalize(self, execution_id, records):
        """Verify every advertised output exists, then fsync, hash and publish each one."""
        entry = self.executions[execution_id]
        by_id = {r["output_id"]: r for r in records}
        folder = self.directory / execution_id
        outputs = []
        for output_id in entry["expected_outputs"]:
            record = by_id.get(output_id)
            if record is None or not Path(record["path"]).is_file():
                raise MissingOutput(output_id)
            with Path(record["path"]).open("rb") as stream:
                head = stream.read(33)
                stream.seek(0)
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                os.fsync(stream.fileno())
            width, height = png.header(head)[:2]
            final = folder / (output_id + ".png")
            Path(record["path"]).replace(final)
            outputs.append(
                {
                    "output_id": output_id,
                    "mime_type": record["mime_type"],
                    "size": final.stat().st_size,
                    "sha256": digest,
                    "width": width,
                    "height": height,
                }
            )
        return outputs

    def run(self, execution_id):
        entry = self.executions[execution_id]
        folder = self.directory / execution_id
        temporary = folder / ".partial"
        state, failure, outputs, measurements = "failed", None, [], None
        try:
            temporary.mkdir(parents=True, mode=0o700)
            records, measurements = self.backend.generate(
                entry["request"],
                temporary,
                lambda kind, data: self.emit(execution_id, kind, data),
                entry["cancel"],
                entry["image_path"],
            )
            outputs = self.finalize(execution_id, records)
            state = "succeeded"
        except Cancelled:
            state = "cancelled"
        except Exception as exc:
            code = (
                "out_of_memory"
                if "OutOfMemory" in type(exc).__name__
                else "execution_failed"
            )
            # A partial batch names the first output that was not produced.
            failure = error(code, exc.output_id if isinstance(exc, MissingOutput) else None)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
            if state != "succeeded":
                shutil.rmtree(folder, ignore_errors=True)
            # The terminal state, its events and the free slot become visible together.
            with self.lock:
                if state == "succeeded":
                    entry["outputs"], entry["measurements"] = outputs, measurements
                    for output in outputs:
                        self.emit(execution_id, "output", output)
                elif failure:
                    entry["error"] = failure
                    self.emit(execution_id, "error", failure)
                    if failure["code"] == "out_of_memory":
                        self.state = "error"
                entry["state"] = state
                self.emit(execution_id, "state", {"state": state})
                self.active = None
                if self.state != "error":
                    self.state = "ready"

    def cancel(self, execution_id):
        with self.lock:
            self.snapshot(execution_id)
            entry = self.executions[execution_id]
            if entry["state"] not in TERMINAL:
                entry["cancel_requested"] = True
                entry["cancel"].set()
            return self.snapshot(execution_id)

    def output_path(self, execution_id, output_id):
        with self.lock:
            snap = self.snapshot(execution_id)
            if snap["state"] != "succeeded":
                raise WorkerError(404, "output_not_found")
            record = next((o for o in snap["outputs"] if o["output_id"] == output_id), None)
            if record is None:
                raise WorkerError(404, "output_not_found")
            return self.directory / execution_id / (output_id + ".png"), record

    def release(self, execution_id):
        with self.lock:
            snapshot = self.snapshot(execution_id)
            if snapshot["state"] not in TERMINAL:
                raise WorkerError(409, "busy")
            shutil.rmtree(self.directory / execution_id, ignore_errors=True)
            del self.attempts[snapshot["attempt_id"]]
            del self.executions[execution_id]


def create_server(worker, token, port=7811):
    if len(token) < 32:
        raise ValueError("Worker token must contain at least 32 characters")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, *args):
            pass

        def respond(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def raw(self, limit):
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= limit:
                raise WorkerError(400 if size <= limit else 413, "invalid_request")
            return self.rfile.read(size)

        def body(self):
            return json.loads(self.raw(4 * 1024 * 1024))

        def events(self, execution_id):
            snap = worker.snapshot(execution_id)
            cursor = int(self.headers.get("Last-Event-ID", "0"))
            if cursor < 0 or cursor > snap["last_sequence"]:
                raise WorkerError(409, "resume_required")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            while True:
                snap = worker.snapshot(execution_id)
                for event in snap["events"][cursor:]:
                    self.wfile.write(
                        f"id: {event['sequence']}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                    )
                    cursor = event["sequence"]
                self.wfile.flush()
                if snap["state"] in TERMINAL:
                    break
                self.wfile.write(b": heartbeat\n\n")
                self.wfile.flush()
                time.sleep(0.25)

        def handle_request(self):
            try:
                url = urlsplit(self.path)
                path = url.path
                if self.headers.get("Origin") is not None:
                    raise WorkerError(403, "browser_origin_denied")
                if path == "/health" and self.command == "GET":
                    return self.respond(
                        503 if worker.state == "error" else 200,
                        {"status": "error" if worker.state == "error" else "ok"},
                    )
                if not hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + token
                ):
                    raise WorkerError(401, "unauthorized")
                if self.command == "GET" and path == "/v1/info":
                    return self.respond(200, worker.info())
                if self.command == "GET" and path == "/v1/capabilities":
                    return self.respond(200, worker.capabilities())
                if self.command == "POST" and path == "/v1/validate":
                    body = self.body()
                    if "bundle" in body:
                        return self.respond(200, worker.validation(body))
                    if "adapters" in body:
                        worker.check(body)
                        strict(body, ("protocol", "adapters"))
                        version = protocol(body)
                        adapters = validate_adapters(
                            body["adapters"], worker.catalog, version
                        )
                        return self.respond(
                            200,
                            {
                                "valid": True,
                                "adapters": [
                                    {"adapter_id": r["adapter_id"], "layout": layout}
                                    for r, _a, _w, _s, layout in adapters
                                ],
                            },
                        )
                    return self.respond(200, worker.validate_execution(body))
                if self.command == "POST" and path in {
                    "/v1/model/load",
                    "/v1/model/unload",
                }:
                    return self.respond(
                        202, worker.control(path.rsplit("/", 1)[1], self.body())
                    )
                if path.startswith("/v1/operations/"):
                    operation_id = path.rsplit("/", 1)[1]
                    with worker.lock:
                        op = worker.operations.get(operation_id)
                        if not op:
                            raise WorkerError(404, "operation_not_found")
                        if self.command == "DELETE" and op["state"] != "running":
                            del worker.operations[operation_id]
                            return self.respond(200, {"released": True})
                        if self.command == "GET":
                            return self.respond(200, op)
                    raise WorkerError(409, "busy")
                if self.command == "POST" and path == "/v1/inputs":
                    if self.headers.get("Content-Type") != "image/png":
                        raise WorkerError(400, "unsupported", "image")
                    data = self.raw(worker.max_input_mb * 1024**2)
                    return self.respond(201, worker.store_input(data))
                input_match = re.fullmatch(r"/v1/inputs/(input_[a-f0-9]{32})", path)
                if input_match:
                    input_id = input_match.group(1)
                    if self.command == "GET":
                        return self.respond(200, worker.input_record(input_id))
                    if self.command == "DELETE":
                        worker.release_input(input_id)
                        return self.respond(200, {"released": True})
                    raise WorkerError(404, "not_found")
                if self.command == "POST" and path == "/v1/executions":
                    return self.respond(202, worker.submit(self.body()))
                if self.command == "GET" and path == "/v1/executions":
                    attempt = parse_qs(url.query).get("attempt_id", [None])[0]
                    with worker.lock:
                        execution_id = worker.attempts.get(attempt)
                        if not execution_id:
                            raise WorkerError(404, "execution_not_found")
                        return self.respond(200, worker.snapshot(execution_id))
                match = re.fullmatch(
                    r"/v1/executions/(exec_[a-f0-9]{32})(/cancel|/events|/outputs/"
                    + OUTPUT_ID
                    + ")?",
                    path,
                )
                if not match:
                    raise WorkerError(404, "not_found")
                execution_id, suffix = match.groups()
                if self.command == "POST" and suffix == "/cancel":
                    snap = worker.cancel(execution_id)
                    return self.respond(200 if snap["state"] in TERMINAL else 202, snap)
                if self.command == "DELETE" and not suffix:
                    worker.release(execution_id)
                    return self.respond(200, {"released": True})
                if self.command == "GET" and not suffix:
                    return self.respond(200, worker.snapshot(execution_id))
                if self.command == "GET" and suffix == "/events":
                    return self.events(execution_id)
                if self.command == "GET" and suffix and suffix.startswith("/outputs/"):
                    with worker.lock:
                        output, record = worker.output_path(
                            execution_id, suffix.rsplit("/", 1)[1]
                        )
                        if not output.exists():
                            raise WorkerError(410, "output_unavailable")
                        with output.open("rb") as stream:
                            self.send_response(200)
                            self.send_header("Content-Type", record["mime_type"])
                            self.send_header(
                                "Content-Length", str(output.stat().st_size)
                            )
                            self.end_headers()
                            while chunk := stream.read(65536):
                                self.wfile.write(chunk)
                    return
                raise WorkerError(404, "not_found")
            except WorkerError as exc:
                self.respond(exc.status, {"error": error(exc.code, exc.field)})
            except (ValueError, KeyError, TypeError):
                self.respond(400, {"error": error("invalid_request")})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            except Exception:
                self.respond(500, {"error": error("internal_error")})

        do_GET = do_POST = do_DELETE = handle_request

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(
        description="Offline local media worker for Media Runtime. Without a subcommand it serves."
    )
    parser.add_argument(
        "--version", action="version", version="media-engine " + WORKER_VERSION
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--port", type=int, default=7811)
    parser.add_argument("--directory", default=".engine")
    parser.add_argument("--catalog")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backend",
        default="diffusers-sdxl",
        choices=["diffusers-sdxl", "owned-sdxl"],
        help="diffusers-sdxl is the reference pipeline; owned-sdxl runs this project's own loop",
    )
    parser.add_argument("--memory-budget-mb", type=int)
    parser.add_argument("--max-retained", type=int, default=128)
    parser.add_argument("--max-storage-mb", type=int, default=1024)
    parser.add_argument("--max-input-mb", type=int, default=32)
    commands = parser.add_subparsers(dest="command", metavar="{convert}")
    from .convert import add_arguments

    add_arguments(
        commands.add_parser(
            "convert",
            help="convert a single-file SDXL checkpoint into a Diffusers-format bundle (offline)",
            description="Convert one single-file SDXL safetensors checkpoint into a complete "
            "Diffusers-format bundle using an approved reference bundle's configuration.",
        )
    )
    args = parser.parse_args(argv)
    if args.command == "convert":
        from .convert import run

        sys.exit(run(args))
    if args.probe:
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "supported_protocols": list(SUPPORTED_PROTOCOLS),
                    "worker_version": WORKER_VERSION,
                    "backend_id": args.backend,
                    "backends": ["diffusers-sdxl", "owned-sdxl"],
                    "commands": ["convert"],
                }
            )
        )
        return
    if not re.fullmatch(r"cuda:\d+|cpu", args.device):
        parser.error("device must be cpu or cuda:N")
    if any(
        v is not None and v < 1
        for v in (
            args.memory_budget_mb,
            args.max_retained,
            args.max_storage_mb,
            args.max_input_mb,
        )
    ):
        parser.error("resource limits must be positive")
    if not args.catalog:
        parser.error("--catalog is required")
    catalog = json.loads(Path(args.catalog).read_text())
    if not isinstance(catalog, dict) or any(
        not isinstance(p, str) or not Path(p).is_absolute() for p in catalog.values()
    ):
        parser.error("catalog must map opaque handles to absolute local directories")
    token = os.environ.get("MEDIA_ENGINE_TOKEN", "")
    if len(token) < 32:
        parser.error("Set MEDIA_ENGINE_TOKEN to at least 32 random characters")
    backend = backend_class(args.backend)(args.device, args.memory_budget_mb)
    # A private per-session directory prevents another process/session's outputs being collected.
    worker = Worker(
        Path(args.directory) / uuid.uuid4().hex,
        backend,
        catalog,
        args.max_retained,
        args.max_storage_mb,
        args.max_input_mb,
    )
    server = create_server(worker, token, args.port)

    def stop(*_):
        with worker.lock:
            worker.state = "error"
            if worker.active:
                worker.cancel(worker.active)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if os.environ.get("MEDIA_ENGINE_SUPERVISED") == "1":
        # The supervisor owns stdin; EOF means its process exited (including a crash).
        def parent_watch():
            sys.stdin.buffer.read()
            stop()

        threading.Thread(target=parent_watch, daemon=True).start()
    print(json.dumps(startup_record(worker, server)), flush=True)
    try:
        server.serve_forever()
    finally:
        deadline = time.monotonic() + 5
        while worker.active and time.monotonic() < deadline:
            time.sleep(0.1)
        if not worker.active:
            worker.backend.unload()
        server.server_close()


def launch_protocol(protocols):
    return "media-worker/1.1" if "media-worker/1.1" in protocols else protocols[-1]


def startup_record(worker, server, **extra):
    """The one stdout line a supervisor reads after launch; no credentials or private paths."""
    record = {
        # `protocol` is the launch-record minor. The record's shape is unchanged since 1.1, and
        # 1.1 supervisors refuse a value they do not know, so it stays media-worker/1.1; callers
        # negotiate from `supported_protocols` (and `highest_protocol`) as they do with /v1/info.
        "protocol": launch_protocol(worker.protocols),
        "supported_protocols": list(worker.protocols),
        "highest_protocol": worker.protocols[-1],
        "worker_version": WORKER_VERSION,
        "backend_id": worker.backend.backend_id,
        "session_id": worker.session_id,
        "port": server.server_port,
        "pid": os.getpid(),
        **extra,
    }
    if worker.speaks_1_2:
        record["device_identity"] = worker.device_identity()
    return record


if __name__ == "__main__":
    main()
