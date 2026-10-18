#!/bin/sh
set -eu

# The desktop base defaults to en_US.UTF-8, which is not generated in Ubuntu.
export LANG=C.UTF-8 LC_ALL=C.UTF-8

mkdir -p /config/settings /config/models /config/gpu-runtimes \
    /config/speech-to-text /config/huggingface /config/cache

# CUDA executables, including Whisper, also need pip's NVIDIA shared libraries.
for directory in /opt/venv/lib/python*/site-packages/nvidia/*/lib; do
    [ -d "$directory" ] || continue
    LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH

cd /data
exec /opt/venv/bin/python /opt/subtitle-sync/subtitle_sync_app.pyw
