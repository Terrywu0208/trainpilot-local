# TrainPilot-Local training image: trainpilot/training:2.41.0-cpu
#
# Base is the pinned Ray CPU image (same digest as the approved-images
# list in src/trainpilot/tools/context.py). The image carries the repo
# training entrypoint. The smoke backend needs nothing beyond the base
# image (pure stdlib + PyYAML, both already in the Ray image), so the
# default build works fully offline after the base pull.
#
# For the real torch fine-tune path, build with --build-arg
# INSTALL_TORCH=1, which installs the CPU wheels of torch,
# transformers and datasets from the PyTorch CPU index. (There is no
# official CPU variant of the pytorch/pytorch image; that is why this
# project builds its own.)
FROM rayproject/ray:2.41.0-py310-cpu@sha256:1c9c04bfb02c9bc9b23f2857a319033e7cb592d71e23663228379f9a8b469629

ARG INSTALL_TORCH=0

WORKDIR /app
COPY training/train.py /app/training/train.py
COPY training/smoke_config.yaml /app/training/smoke_config.yaml
COPY training/corpora /app/training/corpora

RUN if [ "$INSTALL_TORCH" = "1" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
      pip install --no-cache-dir "transformers>=4.44" "datasets>=2.20" "safetensors>=0.4"; \
    fi

# Default: the canonical smoke run. KubeRay overrides the entrypoint
# with the RayJob spec; this CMD is for plain `docker run` checks.
CMD ["python", "/app/training/train.py", "--config", "/app/training/smoke_config.yaml", "--backend", "smoke"]
