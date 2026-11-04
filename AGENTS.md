# Media Engine contributor constraints

Read `engine-spec.md`, `docs/protocol.md`, `docs/decisions.md` and the relevant plan/review item.
The protocol contract, schema, conformance suite, fixture backend and Runtime integration must
advance together. Advertise capabilities only where executable implementations exist.

- No model selection, automatic parameter changes, queue ownership, weight downloads while
  serving, remote code execution or production fixture fallback.
- Verify approved local artifacts and record identity and provenance. Conversion is an explicit
  offline operator command. Unknown fields and unsupported protocol-minor features are errors.
- Preserve single-flight execution, idempotent attempts, cooperative cancellation, output hashes
  and session boundaries. Keep prompts and credentials out of logs, errors and metrics.
- Network serving requires mutual TLS; loopback serving must reject browser origins and hostile
  hosts. Bound request parsing and retained data.
- Record every new Python package and native crate in the dependency ledger. New reference
  families precede additional owned/native family implementations and must state weight licenses.
- Run protocol, conformance and relevant tiny-weight CPU/native oracle tests. Mark CUDA and
  real-weight checks as unrun until evidence exists. Benchmark-gated optimizations remain gated.
- Preserve release asset identities and the shared manual dry-run-first publication workflow.
  Do not publish a production release merely to test release wiring.
