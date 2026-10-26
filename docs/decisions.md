# Implementation decisions and dependency ledger

Runtime-wide decisions for 0.1.0 through 0.3.0 are recorded in Media Runtime's
`docs/decisions.md`. This file starts with the dependency ledger the dependency policy in
[engine-spec.md](../engine-spec.md#dependency-policy) requires, and gains an entry per release
from 0.4.0.

## Dependency policy

Standard library first. A third-party dependency is accepted only when it does something unsafe
or impractical to own, is pinned by checksum, and is isolated in one module so it can be replaced.
Every dependency is listed below with the backend that needs it and the reason it is not owned
code. A dependency without an entry is a defect. No external service is used at any stage.

## Ledger — Python worker (0.3.0, with the planned 0.4.0 split)

| Package       | Needed by                | Why it is not owned code                                        | Planned change                                  |
| ------------- | ------------------------ | --------------------------------------------------------------- | ----------------------------------------------- |
| torch         | reference, owned         | Tensor execution and CUDA kernels                               | Remains until the native engine                 |
| diffusers     | reference, owned         | Reference pipelines; model classes for the owned pipeline       | Owned path imports model classes only           |
| transformers  | reference, owned         | CLIP/T5 model classes and tokenizers                            | Owned path may replace CLIP tokenization later  |
| safetensors   | reference, owned, server | Header parsing and tensor loading                               | Keep                                            |
| numpy         | reference, owned         | Tensor to image conversion                                      | Owned path removes it with the stdlib PNG writer |
| accelerate    | reference                | Diffusers loading helpers                                       | Dropped from the owned path in 0.4.0            |
| Pillow        | reference                | Image decode/encode in Diffusers pipelines                      | Dropped from the owned path in 0.4.0 (`png.py`) |
| peft          | reference                | LoRA application through library layers                         | Not imported on the owned path                  |
| setuptools    | build                    | Packaging                                                       | Keep                                            |

Standard-library-only by design: the HTTP/SSE server (`http.server`), JSON, hashing, TLS in
network mode (`ssl`), WAV (`wave`), ZIP (`zipfile`), PNG writing from 0.4.0 (`zlib`, `struct`).

Optional and not yet accepted: one pinned MP4 encoder wheel for the `video` output format; the
lossless frame archive is the baseline and this entry is added only on evidence of need.

## Ledger — native engine (planned, phase E6)

| Crate   | Why it is not owned code                                   |
| ------- | ---------------------------------------------------------- |
| candle  | Tensor execution; replaced piecewise by owned kernels later |
| cudarc  | Launching owned CUDA kernels                               |
| rustls  | TLS for network mode                                       |
| png     | Deflate-backed PNG encoding                                |

Owned instead of depended on: the HTTP/1.1 and SSE server, the CLIP BPE tokenizer, WAV and ZIP
writers, validation, bundle verification. A `tokenizers` crate is justified only for the T5
tokenizer of the audio family and is recorded when that family reaches the native engine.
