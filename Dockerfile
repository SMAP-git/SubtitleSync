FROM jlesage/baseimage-gui:ubuntu-24.04-v4@sha256:e7c54da3446b4c65b6755bf00a874463c640a1f579f25bed99532051e81649ef

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG CPU_ONLY=0
ENV SUBTITLE_SYNC_CPU_ONLY=${CPU_ONLY}
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    USER_ID=99 \
    GROUP_ID=100 \
    UMASK=0002 \
    SUBTITLE_SYNC_DESKTOP_UPDATES=0 \
    KEEP_APP_RUNNING=1 \
    DISPLAY_WIDTH=1600 \
    DISPLAY_HEIGHT=1000 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    SUBTITLE_SYNC_SETTINGS_DIR=/config/settings \
    SUBTITLE_SYNC_MODEL_CACHE=/config/models \
    HF_HOME=/config/huggingface \
    XDG_CACHE_HOME=/config/cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

# The base defers creation of passwd/group until container initialization.
RUN mkdir -p /config/log \
    && touch /etc/passwd /etc/group \
    && printf 'root:x:0:0:root:/root:/bin/bash\n' >> /etc/passwd \
    && printf 'root:x:0:\nstaff:x:50:\n' >> /etc/group \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl python3 python3-tk python3-venv \
        libgomp1 \
        libgl1 libegl1 ffmpeg mkvtoolnix fonts-dejavu-core \
    && case "$CPU_ONLY" in \
        0) apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers vulkan-tools ;; \
        1) ;; \
        *) echo 'CPU_ONLY must be 0 or 1' >&2; exit 1 ;; \
       esac \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && mkvmerge --version \
    && mkvextract --version

COPY requirements.txt /opt/subtitle-sync/requirements.txt
ARG WITH_CUDA=1
ENV SUBTITLE_SYNC_BUNDLED_CUDA=${WITH_CUDA}
RUN python -m pip install --no-cache-dir -r /opt/subtitle-sync/requirements.txt \
    && test "$CPU_ONLY:$WITH_CUDA" != "1:1" \
    && case "$WITH_CUDA" in \
        1) python -m pip uninstall -y onnxruntime \
           && python -m pip install --no-cache-dir 'onnxruntime-gpu[cuda,cudnn]==1.26.0' ;; \
        0) ;; \
        *) echo 'WITH_CUDA must be 0 or 1' >&2; exit 1 ;; \
       esac \
    && python -c 'import tkinter, fastembed, imageio_ffmpeg, onnxruntime, os; assert ("CUDAExecutionProvider" in onnxruntime.get_available_providers()) == (os.environ["SUBTITLE_SYNC_BUNDLED_CUDA"] == "1")'

ENV APP_NAME="SubSync" \
    APP_VERSION=1.1.1
WORKDIR /opt/subtitle-sync
COPY subtitle_sync_app.pyw subtitle_alignment.py sync_subtitles.py \
    gpu_runtime.py vulkan_embeddings.py manual_sync.py speech_to_text.py \
    speech_dialog.py ui_preferences.py ui_translations.py file_picker.py \
    library_updates.py desktop_updates.py managed_updates.py \
    subtitle_sync_icon.png audio_sync.py audio_sync_dialog.py sync_report.py README.md ./
RUN ln -s /config/gpu-runtimes .gpu-runtimes \
    && ln -s /config/speech-to-text .speech-to-text
COPY --chmod=755 docker/startapp.sh /startapp.sh
# Only the main Tk window is maximized without decorations; dialogs keep close/minimize/maximize.
COPY --chmod=644 docker/main-window-selection.xml /etc/openbox/main-window-selection.xml

EXPOSE 5800/tcp
VOLUME ["/config", "/data"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:5800/ -o /dev/null || exit 1
