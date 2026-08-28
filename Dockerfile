# GPU-enabled image for the video-clipping-agent pipelines.
#
# Requires the host to have the NVIDIA driver + nvidia-container-toolkit
# installed, and the container run with `docker run --gpus all ...` (or
# the cloud provider's equivalent). Nothing in this image substitutes
# for that -- if no GPU is visible at runtime, agent/config.py's
# get_device() silently falls back to CPU rather than failing loudly.
# Check the "device=" value printed in the logs on first run to confirm
# it actually says cuda, not cpu.

FROM python:3.13-slim

# ffmpeg/ffprobe: required by cutter.py, align.py, ingest.py -- not
# pip-installable, must come from the system package manager.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Exact-pinned lock file, not requirements.txt -- see that file's own
# comment for why a Docker image specifically needs tighter pinning.
COPY requirements-docker.lock.txt .
RUN pip install --no-cache-dir -r requirements-docker.lock.txt

COPY agent/ agent/
COPY scripts/ scripts/

# input/ and output/ are meant to be mounted volumes at `docker run`
# time, not baked into the image -- see the run examples in the README.
# Creating them here just means sane default paths exist if no volume
# is mounted (results simply won't persist across container restarts).
RUN mkdir -p /app/input /app/output

ENV VIDEO_AGENT_INPUT_DIR=/app/input
ENV VIDEO_AGENT_OUTPUT_DIR=/app/output
# Drive mounting is a Colab-only mechanism and has no meaning inside a
# container -- left explicit here rather than just omitted.
ENV VIDEO_AGENT_USE_DRIVE=0

ENTRYPOINT ["python", "-m", "agent"]
CMD ["--help"]
