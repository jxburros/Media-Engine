# syntax=docker/dockerfile:1
FROM python:3.12.14-slim-bookworm AS python-base
FROM python-base AS build
# Bound retries and resume interrupted large CUDA wheels before checking hashes.
# These build-only settings do not propagate to the runtime image.
ENV PIP_RETRIES=5 PIP_RESUME_RETRIES=5 PIP_DEFAULT_TIMEOUT=60
WORKDIR /source
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src src
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    python -m pip install --no-cache-dir pip==25.2 uv==0.9.5 && uv export --frozen --no-dev --extra inference --extra inference-audio --extra inference-optimize --extra inference-video --no-emit-project --format requirements-txt > requirements.txt && \
    pip wheel --no-deps --wheel-dir /wheels . && \
    pip wheel --require-hashes --wheel-dir /wheels -r requirements.txt
FROM python-base AS runtime
# The locked torch wheel brings its pinned CUDA userspace libraries. The NVIDIA
# container runtime supplies the host driver when the operator requests --gpus.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 media && useradd --uid 10001 --gid media --create-home media
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/media_engine-*.whl /wheels/*.whl && rm -rf /wheels
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER 10001:10001
WORKDIR /home/media
# Operator mounts catalog and approved read-only weights, and a writable private state directory.
ENTRYPOINT ["media-engine"]
CMD ["--probe"]
