# Subtitle Sync

Synchronize an SRT in any model-supported language against a correctly timed
reference SRT, in the same or a different language, or an embedded SubRip track.
Processing is local; the first analysis downloads the multilingual model.
Original source and reference subtitles are kept unchanged.

## Top bar and interface settings

The top bar shows the app name on the left. On the right are **Speech to text**,
**Analyze and synchronize**, **Cancel**, the **gear menu** and the **version badge**, in that
order. The secondary speech button and primary synchronization button follow the
reference app's light/dark banner colors. The initial Subtitle Sync version is **v1.0.0**.
The gear dropdown is right-aligned with its button; its submenus open to the left
when the window is at the right screen edge, including when maximized.
**Cancel** sits between Analyze and synchronize and the gear, with the same
secondary theme colors as Speech to text. It is enabled during synchronization.
Cancellation waits for the current inference step, then stops without publishing
partial output or replacing existing subtitles. Private ONNX GPU workers are
terminated on cancellation; CPU/Vulkan analysis checks between inference steps.
The header stays visible while the form can scroll on smaller displays; the
initial window height is limited to the available screen height.

The gear offers the same menu categories as Bulk Rename Utility:

- **Language:** English, German, Greek, Estonian, French, Hindi, Hungarian,
  Italian, Japanese, Korean, Dutch, Polish, Portuguese (Brazil), Portuguese,
  Romanian, Russian, Turkish, Simplified Chinese and Traditional Chinese.
  Switching languages immediately translates menus, main controls, workflow
  instructions and common controls in the speech, cue-pairing and GPU windows.
  Paths, subtitle dialogue, backend/model identifiers and spoken-language
  selections are never translated or changed. Detailed technical guidance,
  progress logs and diagnostic messages can still be in English.
- **Theme:** Follow the system, Light or Dark. Popups, text areas and menus follow
  the selected theme. System mode checks Windows' application theme preference or
  the GNOME-compatible desktop color-scheme preference on Linux; systems without
  an available preference use the native light appearance. System mode refreshes
  when the main window regains focus.
- **Help → About Subtitle Sync:** displays this app's name and version.

Language, theme, processing-device choices and Whisper.cpp choices are saved automatically under
`%APPDATA%\SubtitleSync\settings.json` on Windows or
`$XDG_CONFIG_HOME/SubtitleSync/settings.json` on Linux (default:
`~/.config/SubtitleSync/settings.json`). Preference read/write failures are shown
explicitly. Interface changes preserve selected files, devices and manual sync
points.
The main processing device, inference backend, adapter ID and Hearing-impaired
filter level are restored after
restart. The speech window separately remembers the spoken language, model,
device, backend, adapter ID and audio stream index, including across application
restarts. Files and generated-output destinations are not saved. Restored hardware
choices do not guarantee the same GPU is available: recheck adapter IDs if your
hardware or driver configuration changes.

## Starting the app

On Linux, browsing input/output files uses a resizable, maximizable file picker with a
details list. Click **Name**, **Size** or **Modified** to sort; click again to
reverse the order. Folders stay first. Use Up/Home or type a folder path and
press Enter to navigate. This picker also works in the Docker browser desktop.
Windows retains its native file dialogs.

### Portable releases (Windows and Linux x64)

Extract the entire release archive into a writable folder, then run
`Start_Subtitle_Sync.bat` on Windows or
`bash Start_Subtitle_Sync.sh` from a terminal on Linux.
Windows requires PowerShell and `tar.exe` (included with current Windows 10/11).
Linux downloads require `curl`, `tar` and `sha256sum`.

These are **first-launch-download portable releases**, not offline bundles.
The archives contain only app source, requirements, documentation and the
platform launcher. They exclude test videos/subtitles, tests, Python environments,
downloaded packages, models, GPU runtimes, caches and personal preferences.

- A compatible installed Python (3.10+, Tkinter, venv and ensurepip) is reused.
  If none is available, the launcher asks before downloading private
  **CPython 3.12.15**, pinned to python-build-standalone release **20261013** and
  SHA-256-verified. This distribution includes Tkinter and pip, unlike Windows'
  minimal embeddable Python package. Downloads are about 21 MB on Windows and
  33 MB on Linux, extracted under `.python` inside the app folder.
- Python dependencies are installed on first launch into `.venv` on Windows or
  `.venv-linux` on Linux. Later launches check and reuse them. First setup needs
  internet and sufficient disk space; inference models and optional GPU runtimes
  are downloaded separately when needed.
- The release's `PORTABLE` marker makes the launchers keep app preferences in
  `.config/settings.json` and the matching model cache in `.model-cache`, inside
  the extracted app folder. Speech models and GPU runtimes already live there.
  Existing user settings from a source installation are not copied into releases.
  Dependency tools can still use their normal OS download caches.
- Linux still needs a graphical desktop and compatible system libraries
  (including X11/OpenGL). The launcher gives package guidance if needed; it never
  runs sudo or installs OS packages/drivers. Private Python targets glibc Linux
  x86_64, not Alpine/musl. Windows private Python targets x64, not ARM64.
- Extract into the intended final location before first launch. Do not copy a
  generated virtual environment between machines or operating systems. To move
  to a new computer/location, extract a fresh archive and copy only desired data,
  downloaded models and preferences; let its launcher recreate dependencies.
- Keep the launcher terminal open while the app runs so startup errors remain
  visible. No system-wide Python installation or PATH changes are performed.

Build clean archives with `python3 build_portable.py`. Its explicit allowlist
produces Windows ZIP and Linux tar.gz archives plus `SHA256SUMS.txt` in `releases`.
Unraid/Docker setup is described below; it is separate from these portable archives.

- Windows: run `Start_Subtitle_Sync.bat`. A suitable installed Python is reused;
  otherwise you can approve a private runtime download.
- Linux: run `bash Start_Subtitle_Sync.sh`. System Python 3.10+, venv and
  Tkinter are preferred; a private x64 runtime download is offered if needed.
  On Debian/Ubuntu, system prerequisites include `python3-venv python3-tk`.

The launchers use separate `.venv` (Windows) and `.venv-linux` (Linux) environments.
Both platforms have the same UI, including the hearing-impaired filter and initial
sync point. Embedded references use MKVToolNix for Matroska when available
(`mkvmerge` and `mkvextract` on PATH; Windows installation paths are also checked).
Otherwise the app uses FFmpeg, including its bundled executable, to inspect and
extract supported text subtitle tracks. Neither a separate FFmpeg installation
nor MKVToolNix is required.

### Desktop library updates and rollback

Desktop startup checks for newer **fastembed**, **imageio-ffmpeg** and
**onnxruntime** versions and installed app-managed runtimes/models at most once
every 24 hours, in the background. Checks also repeat daily if the app stays open.
Initial
setup still installs the tested versions from `requirements.txt`. A successful
check with available updates opens **gear -> Help -> Library updates**; no
packages are installed without your confirmation. You can also open this window
to check manually, see running versions, install updates or restore the previous
library environment.

**Install updates** creates a separate Python environment in
`.library-updates/linux` or `.library-updates/windows` inside the app folder.
It installs the latest stable releases compatible with your Python, including
their required dependencies, then checks package consistency and app imports.
If any step fails, the current selection stays unchanged and the error is shown.
An import check cannot guarantee a newer library works correctly for every
subtitle or device; keep the previous environment until you have tested it.
Package checks/installation use pip's configured package index (PyPI by default).
Network failures are shown, not reported as "up to date"; automatic checks are
retried on a subsequent launch after 24 hours, or manually at any time.

After installation or restoration, close the app and reopen it **using the
desktop launcher** to use the selected libraries. The working environment is not
modified while the app is running. **Restore previous versions** switches back
to the exact previous environment without downloading packages. The original
environment is kept, and older managed environments are retained on disk.
App preferences remain shared; updating desktop libraries does not change the
separately selected models or private GPU runtimes. An update operation requires a writable app folder and may use
substantial disk space.

If an update prevents the GUI from opening, restore from a terminal:

```sh
# Linux
bash Start_Subtitle_Sync.sh --restore-libraries
```

```bat
:: Windows
Start_Subtitle_Sync.bat --restore-libraries
```

The recovery option switches to the previous library environment and starts the app.
Python itself and system OS packages, drivers, CUDA and ROCm installations are
never updated by this feature. `SUBTITLE_SYNC_DESKTOP_UPDATES=0`
disables the desktop check and update controls. Directly running the GUI with
an arbitrary Python does not activate a selected environment; use the launcher.

#### Installed runtime and model versions

The **Installed runtimes and models** selector in the same update window checks
only components already installed for this platform:

- Private NVIDIA, DirectML and OpenVINO environments, including their GPU ONNX
  distribution, fastembed and installed NVIDIA CUDA/cuDNN Python packages.
- The llama.cpp Vulkan embedding executable and its multilingual GGUF model.
- Installed Whisper.cpp CPU, Vulkan and CUDA executables.
- Downloaded Whisper model variants and the cached multilingual ONNX embedding
  model (Hugging Face snapshots, including older cache-name casing, or the legacy
  fastembed model folder).

Select a component, then **Install updates** or **Restore previous versions**.
Each component has its own active/previous selection. Updates use separate
directories under `.component-updates/linux` or `.component-updates/windows`;
original files are retained. These selections are used by GPU workers, speech
transcription and embedding inference, not just shown in the update window.
Restart after a change before continuing work. Model updates retain the same
model family and variant; they do not switch a `tiny` Whisper model to a larger
one or replace the alignment algorithm.

Runtime checks query the existing upstream repository and match the installed
OS/backend asset. Official llama.cpp and whisper.cpp numbered `b...` binary
builds are checked on their existing build channel even when GitHub marks them
as prereleases; unrelated prereleases are not selected. A newer source-only
release is not a usable binary update. Downloads require upstream SHA-256 asset
digests; models use immutable Hugging Face revisions and file checksums.
Archives must contain the expected executable, which is tested with `--version`.
Embedding snapshots are also tested for finite 384-dimensional CPU embeddings.
Whisper/GGUF models are checksum- and format-checked. GPU environments must
expose the required provider; this does not replace testing inference with your
actual GPU, driver and adapter ID.

If no matching downloadable asset or checksum is available, or an upstream check
fails, the window reports it and leaves the working version alone. AMD Linux
MIGraphX wheels depend on the installed OS/ROCm stack: newer packages are reported
when discoverable, but automatic replacement is not offered. Use the existing
GPU setup guidance for those custom wheels. No system installers are run.

If a component change prevents normal startup, restore all components that have
a previous version without needing the GUI:

```sh
bash Start_Subtitle_Sync.sh --restore-components
```

```bat
Start_Subtitle_Sync.bat --restore-components
```

This recovery operation does not switch desktop library environments; use
`--restore-libraries` separately if needed. Older versions remain on disk; no
runtime, model or environment history is deleted automatically.

**Docker/Unraid is excluded:** there are no library update checks, notifications,
or install/restore controls in containers. Bundled library updates require a
new image, built with revised dependency pins. This feature does not change
`requirements.txt` or build/release anything automatically.

## Unraid / Docker (Linux x64)

### CPU-only image (recommended small download)

For CPU users, choose `subtitle-sync:1.0.0-cpu`; for GPU users choose the full
`subtitle-sync:1.0.0` image. The earlier slim variant is discontinued.

CPU-only includes Python, Tkinter/browser desktop, CPU ONNX Runtime, FFmpeg and
MKVToolNix. It excludes the optional Vulkan driver packages and NVIDIA CUDA/cuDNN
libraries. Only CPU processing is offered in matching and speech-to-text; GPU
runtime installation is disabled. CPU models and the CPU Whisper executable
still download on demand into persistent `/config`.

```sh
docker build --platform linux/amd64 --build-arg WITH_CUDA=0 --build-arg CPU_ONLY=1 -t subtitle-sync:1.0.0-cpu .
docker save subtitle-sync:1.0.0-cpu | gzip > SubtitleSync-1.0.0-docker-amd64-cpu.tar.gz
gzip -dc SubtitleSync-1.0.0-docker-amd64-cpu.tar.gz | docker load
```

Use [unraid/subtitle-sync-cpu.xml](unraid/subtitle-sync-cpu.xml), or change an
existing container's Repository to `subtitle-sync:1.0.0-cpu` after loading it
and Apply. Remove GPU Device entries and `--gpus=all` / `--device=/dev/dri:/dev/dri`
from Extra Parameters; retain `--shm-size=1g`. Keep existing `/config` and `/data`
mappings. Saved GPU selections are ignored in this variant and CPU is selected.
Do not run two instances against the same appdata directory.
Compose: `docker compose -f compose.yml -f docker/compose.cpu.yml up -d --build`.
No NVIDIA plugin or GPU passthrough is needed.

The image uses the same pinned `jlesage/baseimage-gui` browser-desktop base as
[the Subtitle Edit Unraid reference](https://github.com/madsmap-beep/subtitle_edit_unraid_nvidia).
Open port **5800** in a browser to use the normal Tkinter app; no desktop Python,
X server or installation script is needed on the client or at container startup.
Python, Tkinter, FFmpeg, MKVToolNix, Python dependencies, Mesa Vulkan drivers and
CUDA 12/cuDNN userspace libraries are installed **when the image is built**.
Models and app-specific Vulkan/Whisper executables are still downloaded and
checksum-verified on demand, and persist in `/config`.
The all-vendor image is substantially larger than the portable archives because
it includes NVIDIA userspace libraries even on CPU/AMD/Intel hosts.
The current all-vendor build is about 2.4 GB compressed and 7 GB in Docker image
storage (reported sizes vary with Docker's layer/compression accounting).
NVIDIA libraries alone occupy roughly 2.8 GB uncompressed, plus about 470 MB for
ONNX Runtime. The desktop, Python, media tools and Mesa libraries add to that.
Downloaded inference models use additional space in `/config`, not in the image.

MKVToolNix (`mkvmerge` and `mkvextract`) is included. The container uses the
built-in `C.UTF-8` locale, including for child media tools. Older images inherited
an unavailable `en_US.UTF-8` locale from the desktop base; that could crash
MKVToolNix with `locale::facet::_S_create_c_locale name not valid`.
Rebuild/load the corrected image and recreate the container with the existing
`/config` mount. As an immediate workaround on an older image, add an Unraid
Variable named `LC_ALL` with value `C.UTF-8`, then Apply to recreate the container.

### Build and install

Two images are available:

| Image | NVIDIA libraries | Best for |
|---|---|---|
| `subtitle-sync:1.0.0` (full GPU) | Included | CPU, NVIDIA CUDA and AMD/Intel Vulkan GPU/APU users |
| `subtitle-sync:1.0.0-cpu` | Not included; GPU disabled | CPU-only users wanting the smaller download |

Both include the resizable/maximizable sortable file picker, synchronization
Cancel button, cached-file download prompt fix, UTF-8 locale fix and single-pass Whisper
zero-duration cue repair. Models and app-specific Whisper executables still
download on demand. GPU host drivers/plugin and passthrough remain required for
GPU use. Do not run two containers simultaneously against the same appdata.

For the full image:

```sh
docker build --platform linux/amd64 -t subtitle-sync:1.0.0 .
# Optional: transfer the image to a different machine.
docker save subtitle-sync:1.0.0 | gzip > SubtitleSync-1.0.0-docker-amd64.tar.gz
```

On Unraid, load the image with
`gzip -dc SubtitleSync-1.0.0-docker-amd64.tar.gz | docker load`.
Copy [unraid/subtitle-sync.xml](unraid/subtitle-sync.xml) into
`/boot/config/plugins/dockerMan/templates-user/my-subtitle-sync.xml`, then use
**Docker -> Add Container -> Template -> SubtitleSync**. Set your media share,
appdata location, Web UI host port and optional GPU options before applying.
The template refers to the **local** `subtitle-sync:1.0.0` image: it is not a
published Docker Hub/GHCR image and cannot be pulled from a registry.
To update it, rebuild/load the new image and recreate the container while keeping
the same `/config` mapping; Unraid's registry update check cannot update a local image.

The build context is an explicit allowlist in `.dockerignore`, and the image
copies only required application files: no test videos/SRTs, local environments,
downloaded runtimes/models, preferences or portable release archives are included.

For a normal Docker host, set `SUBTITLE_SYNC_CONFIG` and `SUBTITLE_SYNC_MEDIA`
to the desired host directories, and `USER_ID`/`GROUP_ID` to their owner.
On Unraid the defaults are `99:100` (nobody:users). Example CPU deployment:

```sh
SUBTITLE_SYNC_CONFIG=/mnt/user/appdata/subtitle-sync \
SUBTITLE_SYNC_MEDIA=/mnt/user/media docker compose up -d --build
```

Access `http://SERVER-IP:5800`. Files are selected inside `/data`. Preferences,
matching/speech models, caches and downloaded GPU executables persist under
`/config`. Container recreation must reuse that mount. Only `/config` ownership
is managed automatically; `/data` must already be writable by the configured
user/group for saving subtitles. No privileged container or host-wide filesystem
mount is needed.

Both images default to `UMASK=0002`: synchronized and speech-to-text subtitles
are saved with `-rw-rw-r--` (`664`) permissions on normal Linux filesystems.
To change this, add/edit the `UMASK` variable in Unraid and Apply (for example,
`0022` gives `644`). Host filesystem ACLs can also affect permissions.
The sync-output permissions fix requires rebuilding/loading the corrected image
and recreating the container with the same `/config` mapping. It applies to newly
saved outputs, including replacements; existing files are not changed retroactively.

### GPU / APU passthrough

| Hardware | Host/container setup | App choices |
|---|---|---|
| CPU | No devices or NVIDIA runtime required | CPU |
| AMD GPU or AMD APU | Host AMD kernel driver; map `/dev/dri` | AMD or AMD APU (integrated), Recommended/Vulkan |
| Intel GPU or integrated graphics | Host i915/xe driver; map `/dev/dri` | INTEL or INTEL integrated GPU, Recommended/Vulkan |
| NVIDIA | Unraid NVIDIA Driver plugin / NVIDIA Container Toolkit; `--gpus=all` | NVIDIA, Recommended/CUDA; Vulkan is an alternative when exposed by the driver |

AMD/Intel in Compose:

```sh
docker compose -f compose.yml -f docker/compose.dri.yml up -d --build
```

NVIDIA in Compose (requires Compose 2.30+ and configured NVIDIA Container Toolkit):

```sh
docker compose -f compose.yml -f docker/compose.nvidia.yml up -d --build
```

Both override files can be supplied together on a host with mixed GPUs.
In Unraid, use **Extra Parameters** in Advanced View:

- CPU only: `--shm-size=1g`
- AMD/Intel GPU or APU: `--shm-size=1g --device=/dev/dri:/dev/dri`
- NVIDIA: `--shm-size=1g --gpus=all`
- Mixed NVIDIA and AMD/Intel: `--shm-size=1g --gpus=all --device=/dev/dri:/dev/dri`

For NVIDIA, set `NVIDIA_VISIBLE_DEVICES` to the GPU UUID from the NVIDIA Driver
plugin (or `all`). Only add the DRI argument if the host has `/dev/dri`.
Do not leave a blank Device entry in the Unraid form: DockerMan emits
`--device=''`, which fails with `bad format for path`. If using an older template,
remove its **AMD/Intel GPU or APU** Device entry entirely rather than clearing its
value, then use Extra Parameters as above.
A GPU bound to VFIO for a VM is not
available to this container.

The desktop base automatically adds normal non-root DRI device groups to the
application user. If needed, enter additional numeric groups in `SUP_GROUP_IDS`;
do not solve permission errors by using privileged mode or running the app as root.
Host kernel drivers **cannot be installed into the app folder/container**.
The image supplies userspace libraries, not Unraid kernel drivers. No ROCm
installation or `/dev/kfd` mapping is required for the recommended AMD Vulkan path.
Advanced MIGraphX/OpenVINO paths are not preinstalled in this image; use Vulkan.
GPU/APU compatibility varies; not every card/driver supports the required Vulkan
or CUDA features. The APU option uses its integrated GPU, not an NPU.

Select the device **separately** in the main window and Speech to text window.
Use **GPU setup -> Test GPU** for matching, and device discovery/testing in the
speech window before a full job. For NVIDIA, matching's ONNX CUDA provider is
already installed in the main Python environment; no extra private ONNX install
is needed. A listed provider is not proof of successful GPU inference.
Whisper/Vulkan executables and models are downloaded through the app's setup.
GPU errors remain visible and CPU mode stays available; a software Vulkan
adapter such as llvmpipe/lavapipe is not hardware acceleration.

Useful diagnostics (replace `CONTAINER` with the container name/ID):

```sh
docker exec CONTAINER vulkaninfo --summary
docker exec CONTAINER nvidia-smi             # NVIDIA hosts only
docker logs CONTAINER
```

Keep the Web UI on a **trusted LAN**: it is a remote desktop with access to
mounted media. Do not forward port 5800/5900 to the Internet. For remote use,
use a VPN or authenticated TLS reverse proxy. The optional `VNC_PASSWORD`
(maximum eight characters) is not transport encryption. Direct VNC port 5900
is not published by the supplied Compose configuration/template.

## Selecting the initial sync point

1. Choose **Source SRT (to synchronize)** and a **Reference SRT** whose timestamps
   already match the video. With an external reference SRT, the video is optional.
2. Alternatively, leave Reference SRT blank and choose a video. The app checks
   embedded subtitles automatically, even before a source SRT is selected:
   - No usable embedded text subtitles: **Speech to text** opens automatically
     so you can generate a reference SRT.
   - One usable track: it is selected automatically.
   - Multiple usable tracks: a popup lets you choose the reference track.
   All available SubRip languages are offered; English is not preferred.
   The FFmpeg path also converts supported ASS/SSA, WebVTT and MP4 text tracks
   to SRT. Image-only subtitles cannot be used without OCR; use speech to text
   if no usable text track is available.
   An already-filled reference SRT field bypasses this automatic check.
3. Click **Select matching cues...**. A popup displays both subtitles side by side
   with cue numbers, timestamps, searchable dialogue and full-text previews.
4. Select one cue in each subtitle that represents the **same dialogue moment**,
   then click **Use selected pair**. The source cue can be anywhere in the file.
   Its target start time comes directly from the reference cue; the app never
   replaces your choice with an automatic first-cue guess.
5. Choose an output file and click **Analyze and synchronize**.

The selected source cue is a fixed anchor. Other reliable matches are found both
before and after it; the reference timeline is not shifted to compensate for
the source's existing offset. Changing either input requires selecting the pair
again. Input-file changes during loading/analysis are detected and prevent output
from being written using a stale selection. Saving over either input is prohibited.
Cancelling cue selection does not start synchronization.

## No reference SRT? Generate one from speech

Click **Speech to text...** in the main window. This uses local
[Whisper.cpp](https://github.com/ggml-org/whisper.cpp), following Subtitle Edit's
Whisper CPP/Vulkan approach, and works independently of subtitle matching.
You can also generate a standalone transcript without selecting a source SRT.

1. Choose a video/audio file and a **separate generated reference SRT** destination.
2. Select the video's **spoken language**, or **Auto detect**. The output stays in
   that language: it is not translated to English. For videos containing multiple
   audio tracks, enter the desired zero-based **audio stream index** (0 = first
   audio stream, 1 = second audio stream, regardless of video/subtitle stream IDs).
3. Choose **tiny**, **base**, **small**, **medium**, **large-v3**, or
   **large-v3-turbo**. English-only `.en` variants are also available for
   tiny/base/small/medium and require **English**, not Auto detect.
   Cantonese's dedicated language code requires a large-v3-family model.
4. Select **CPU**, **NVIDIA**, **AMD**, **INTEL**, **AMD APU (integrated)**,
   or **INTEL integrated GPU**. **Recommended** uses CPU, CUDA for NVIDIA,
   and Vulkan for AMD/Intel GPUs and APUs. NVIDIA can also use Vulkan explicitly.
   These settings do not change the main window's subtitle-matching settings.
5. For GPU/APU use, click **Detect GPUs / APUs** to run a short Whisper test and
   display actual adapter IDs/names. Enter the desired **GPU adapter ID**.
   IDs are specific to Whisper, not necessarily the same as the matching engine.
   Vendor/APU labels do not automatically select or filter the physical adapter.
6. Click **Generate reference SRT**. Approve the first-time downloads.
   **Download runtime / verify model** is also available to prepare files before
   transcription. Later runs reuse cached files and can work offline.
7. The generated SRT is saved and automatically selected as the main window's
   reference. Close the speech window, select your source SRT, then use
   **Select matching cues...** to choose the initial sync pair as before.

### Downloads and hardware requirements

- The app downloads pinned, SHA-256-checked Whisper binaries/models under
  `.speech-to-text` beside the app. Model downloads range from about 78 MB
  (tiny) to 3.1 GB (large-v3); large-v3-turbo is about 1.6 GB. Runtime downloads
  range from roughly 5-23 MB for CPU/Vulkan to 259-685 MB for CUDA.
- CPU and Windows CUDA packages come from the official Whisper.cpp **b5454**
  release (v1.9.5). Linux CUDA and Windows/Linux Vulkan packages come from
  [Subtitle Edit's whispercpp-195 release](https://github.com/SubtitleEdit/support-files/releases/tag/whispercpp-195).
  CPU supports Windows/Linux x64 and ARM64; the bundled GPU packages support x64.
  Older Linux distributions may need newer system libraries to run these binaries.
- FFmpeg is used from PATH when available; otherwise the launchers install
  `imageio-ffmpeg`, which supplies a private FFmpeg executable on supported
  Windows/Linux platforms. Audio is decoded to temporary mono 16 kHz PCM.
  Initial audio delays are retained relative to the media start. Temporary files
  are removed after completion, failure or cancellation.
  Whisper splits segments at word boundaries with an 84-character target to
  avoid paragraph-sized subtitle cues. Download confirmation is shown only for
  missing/incomplete runtime or model files, not every time cached files are used.
  GPU detection checks the tiny model it actually uses, independently of the
  transcription model selection. Cached models are still checksum-verified;
  damaged models are downloaded again.
  Whisper runs once. If word splitting produces zero-duration cues (seen with
  CUDA), each such word is joined to the nearest neighbouring line using only
  Whisper's own timestamps; empty cues (no dialogue) are removed. The log
  reports how many were repaired. Reversed timings, or a transcript with only
  zero-duration cues, are rejected with the offending cue named, and existing
  output remains unchanged.
- Your **existing system graphics driver** is used. The app does not download or
  replace kernel/display drivers or require administrator access. Vulkan needs
  a working Vulkan loader/driver; CUDA needs a compatible NVIDIA driver and may
  report missing host runtime libraries. Resolve the reported dependency or use
  Vulkan/CPU; GPU failures are not silently accepted as CPU transcription.
- Start with **tiny/base on older GPUs or APUs**. Larger models may exhaust VRAM
  or shared memory. Flash attention is disabled for compatibility with older
  Vulkan hardware. GPU inference still uses CPU resources for some operations.
- Real CPU and AMD GPU (Linux RADV Vulkan) transcription have been verified.
  Windows, CUDA and other GPUs/APUs need testing on the respective hardware.

The window shows progress and an engine log, stays responsive, and supports
**Cancel**. Cancellation can take up to the network timeout during a stalled
download. Existing output requires explicit confirmation; inputs and the selected
synchronized-output path are protected. An SRT is published only after successful
transcription and validation; failures/cancellation do not replace it.

Whisper can mishear names, omit dialogue or hallucinate text during silence.
Review generated text and timestamps before synchronizing. Spoken-language
recognition coverage is broader than the matching model's language coverage;
a selectable speech language does not guarantee reliable cross-language matching.
Source cue durations in the final synchronized SRT are still preserved, subject
to the existing overlap/negative-timestamp corrections.

## Languages and contextual matching

The multilingual MiniLM model covers roughly 50 languages, including English,
Portuguese, Spanish, French, German, Arabic, Chinese, Japanese, Korean and Russian.
No Portuguese/English translation pipeline or language setting is required.
Unicode dialogue is retained when building candidates; a Latin alphabet is not
required. Support and quality vary by language, especially for languages absent
from the model's training set.

Matching compares semantic embeddings, not literal words. It evaluates short
multi-cue phrases **on both sides** (up to five nearby cues and nine seconds;
individual longer cues, including Whisper segments, are retained) to
accommodate different sentence splits, and compares surrounding dialogue context
to help disambiguate paraphrases and repeated expressions. Context and phrase
similarity are combined with the initial offset, a local time search and
chronological constraints. Cross-language character-length penalties are not used.
Unreliable matches are rejected rather than treated as exact translations.

This is semantic similarity, not a reasoning/translation model: idioms, heavily
rewritten scenes, missing dialogue or different edits can still defeat automatic
matching. Review the output, especially where there are few confident matches.
Hearing-impaired filtering uses language-independent brackets/music markers plus
an English sound-description vocabulary; the latter is not exhaustive across
languages. The user-selected anchor remains fixed even when its dialogue is
paraphrased or would be excluded by that filter.

## CPU and graphics cards

Select **CPU**, **NVIDIA**, **AMD**, **INTEL**, **AMD APU (integrated)**, or
**INTEL integrated GPU** in Processing device. CPU is the
default and works with the standard installation. GPUs accelerate model inference;
subtitle parsing, matching and retiming still use the CPU.

Leave **Inference backend** set to **Recommended** for these defaults:

| Selection | Windows backend | Linux backend |
| --- | --- | --- |
| CPU | ONNX Runtime CPU | ONNX Runtime CPU |
| NVIDIA | CUDA | CUDA |
| AMD | Vulkan | Vulkan |
| INTEL | Vulkan | Vulkan |
| AMD APU (integrated) | Vulkan | Vulkan |
| INTEL integrated GPU | Vulkan | Vulkan |

Integrated selections use the GPU built into an AMD or Intel processor, not CPU
inference. They share the Vulkan runtime; enter
the integrated adapter's ID if a discrete GPU is also present. Selecting an
integrated option does not automatically discover or choose that adapter.
Intel HD/UHD, Iris and integrated Arc support depends on the processor generation,
driver and Vulkan features. A successful GPU test is required to establish whether
the selected hardware works with this runtime.

GPU support depends on your card, drivers, Python version and installed runtime;
having a vendor's GPU alone is not sufficient. The UI reports whether the required
runtime is installed, not whether a physical GPU or its drivers have been validated.
Analysis reports an error if the selected backend cannot initialize. It does not
silently switch to CPU. Vulkan startup checks that model layers are offloaded to
GPU; CPU tokenization and some operations/buffers remain normal.

**GPU adapter ID** is zero-based and refers to the backend's device ordering.
Use **Detect GPUs** in Vulkan setup to see actual device names and IDs. These IDs
are backend-specific: do not reuse a Vulkan ID blindly for CUDA or DirectML.
Vulkan enumerates all vendors; choose the actual GPU/APU you want, not just the
first adapter. CUDA and MIGraphX enumerate their supported devices. OpenVINO uses `GPU.0`,
`GPU.1`, etc. DirectML enumerates **all DXGI adapters**, not only the selected
vendor: AMD and INTEL both use DirectML on Windows, so enter the adapter ID for
the actual card you want. The vendor selection does not detect or filter cards.
Use your runtime's device enumeration tools to determine IDs on multi-GPU systems.

### Easy Vulkan setup: AMD, Intel, or NVIDIA (Windows and Linux)

1. Select your GPU/APU under **Processing device**. For AMD/Intel, leave
   **Inference backend** on **Recommended**. For NVIDIA, select **Vulkan**
   explicitly if you prefer it to CUDA.
2. Click **GPU setup / drivers...**, then **Install / repair Vulkan support**.
3. Approve downloading about 275 MB. The app downloads the official llama.cpp
   Vulkan binaries and a community GGUF F16 conversion of the **same multilingual
   MiniLM model family** used by CPU/CUDA. Both downloads are pinned and
   SHA-256-verified; files stay under `.gpu-runtimes` in the app folder.
4. The app detects hardware adapters. Select yours in the list; its Vulkan ID is
   copied to the main window. Click **Test GPU**.
5. When the test passes, close setup and synchronize as normal. No app restart,
   ROCm installation, CUDA toolkit or Python GPU package is needed for Vulkan.

**If Vulkan already works, do not reinstall graphics drivers.** The app uses the
existing system Vulkan driver/loader, like other Vulkan applications. A successful
Whisper.cpp Vulkan run is encouraging but not a guarantee: this is a different
runtime/model with its own required Vulkan features. If detection fails, check the
existing driver/loader (Linux: `vulkaninfo`); only update/install system components
when needed. The app never modifies system drivers or runs elevated commands.

The Vulkan path was tested on an AMD GPU with Linux RADV, including real
multilingual embeddings and subtitle alignment. The earlier
ROCm restriction applies only to the optional ONNX/MIGraphX path, not Vulkan.
Other cards/drivers and Windows still need their own **Test GPU** check.

The Vulkan runtime is [llama.cpp](https://github.com/ggml-org/llama.cpp/releases/tag/b11707)
with embedding support, not Whisper.cpp speech recognition. Its local server binds
only to `127.0.0.1`, uses a random API key and is stopped after embeddings are
generated or an error occurs. Subtitle text is not sent to an internet service.
CPU and CUDA retain the existing FastEmbed/ONNX implementation. Vulkan's F16 GGUF
conversion/tokenizer can produce slightly different similarities and matches, so
review important synchronization results. Both paths use normalized 384-dimensional
vectors, mean pooling and a 128-token input limit.

Packages are available for Windows/Linux x64 and ARM64. Linux binaries are upstream
Ubuntu builds and may require newer glibc/system libraries than an older
distribution provides. A working Vulkan driver alone does not guarantee binary
compatibility. Downloads/installation failures are shown explicitly; CPU remains
available. Internet access is needed for initial setup only.

### CUDA and advanced ONNX GPU setup

NVIDIA **Recommended** uses CUDA. To retain the previous AMD/Intel inference
backends, select **ONNX (advanced)** explicitly:

| Selection | Windows ONNX backend | Linux ONNX backend |
| --- | --- | --- |
| NVIDIA | CUDA | CUDA |
| AMD / AMD APU | DirectML | MIGraphX (ROCm) |
| Intel / Intel integrated | DirectML | OpenVINO GPU |

1. Start the app normally and select your GPU vendor under **Processing device**.
2. Click **GPU setup / drivers...**.
3. Use **Official driver guide** only if your system driver needs installation/updating.
   Linux users should follow the instructions for their distribution. Restart the
   computer if the driver installer asks. The app never runs elevated commands,
   changes system drivers, or downloads driver installers from unofficial sites.
4. Click **Install / repair GPU support** and approve the download. The app creates
   a private Python environment under `.gpu-runtimes` in the app folder, downloads
   the runtime and its dependencies, and then runs a real subtitle-model inference
   test on the selected adapter. Keep the app open until it finishes.
5. After a successful test, close the setup window and synchronize as normal.
   No app restart or terminal commands are needed. Use **Test GPU** to retest
   after a driver update or when changing the adapter ID.

Automatic packages:

- **NVIDIA Windows/Linux:** `onnxruntime-gpu[cuda,cudnn]>=1.21`, including packaged
  CUDA/cuDNN libraries, preloaded before model initialization. This does **not**
  include the system NVIDIA driver. Current package versions may require a very
  recent driver or CUDA generation; follow the error output and official guide.
- **AMD/Intel Windows:** `onnxruntime-directml`; requires DirectX 12-compatible
  hardware and its system graphics driver. Both vendors share the DirectML environment.
- **Intel Linux:** `onnxruntime-openvino`; requires Intel GPU drivers and device
  permissions configured according to Intel's guide.
- **AMD Linux ONNX:** guided manual setup (below); use Vulkan to avoid ROCm.

GPU packages may require several GB of disk space. The setup log displays download,
installation and test errors. Your Python version and architecture must have
compatible wheels; if not, create the app's base environment with a Python version
supported by the selected runtime, or continue using CPU. Setup uses the same
Python version as the running app; it does not install Python or change OS packages.

The base environment and CPU mode remain unchanged. Each backend has a separate
environment, and subtitle analysis automatically runs in that environment in a
child process. Failed installation or testing does not silently switch analysis to
CPU. Select CPU explicitly if desired. A runtime marked installed is not proof of
working hardware: **Test GPU** runs inference and checks the requested provider
remains active. It does not guarantee every operation was executed on GPU.

### Advanced AMD Linux ONNX: ROCm/MIGraphX

Select **AMD**, set **Inference backend** to **ONNX (advanced)**, then open
**GPU setup / drivers...**. The Linux setup panel lets you choose:

- **Official AMD driver / ROCm**: guidance for the system driver and compute stack.
- **MIGraphX inference**: guidance for the ONNX Runtime backend used by this app.
  MIGraphX is also an official AMD project, not an alternative system driver.
  These are complementary setup stages, not interchangeable inference backends.

Select or type your **exact GPU model**, **distribution**, and **distribution release**.
The distribution/release are initially detected from Linux's OS-release information
when available and remain editable. The GPU suggestions include Radeon, Ryzen and
Instinct models; unlisted models can be typed. Suggestions do **not** indicate support.
The expanded list includes RX 400/500, Vega,
RX 5000/6000/7000/9000 and older R7/R9 cards. Selecting **AMD APU (integrated)**
shows an APU-focused model list instead.

The panel shows selection-specific guidance, package-manager-family cautions, and
copyable app-environment commands for the MIGraphX stage.

Click **Compatibility matrix** before installing. **Open selected setup guide**
opens the official ROCm selector or MIGraphX documentation for the selected stage;
repeat your exact GPU/OS selection on AMD's site (website fields are not prefilled).
**ONNX Runtime guide** explains the required MIGraphX execution-provider build.
The app does not verify support or choose a wheel based on the selections, and
does not substitute another distribution's packages. Installation stays manual;
**Install / repair GPU support** remains disabled for AMD Linux.

After installing the required system driver/ROCm components using those guides:

1. Create `.gpu-runtimes/linux-amd` with a compatible Python:
   `python3 -m venv .gpu-runtimes/linux-amd`.
2. Install the app dependencies:
   `.gpu-runtimes/linux-amd/bin/python -m pip install -r requirements.txt`.
3. Uninstall the CPU runtime:
   `.gpu-runtimes/linux-amd/bin/python -m pip uninstall -y onnxruntime`.
4. Install AMD's compatible ONNX Runtime wheel/build with
   `MIGraphXExecutionProvider` into that environment using its Python and AMD's
   instructions. Do **not** install the standard PyPI `onnxruntime-gpu` for AMD.
5. Select AMD and **ONNX (advanced)** in the app, open **GPU setup / drivers...**,
   then click **Test GPU**.
   A successful test registers the environment for subsequent analysis.

If your GPU, distribution or Python version is unsupported, try Vulkan or use CPU
rather than installing mismatched ROCm packages.

See the official [ONNX Runtime execution provider documentation](https://onnxruntime.ai/docs/execution-providers/)
for current driver, runtime and platform compatibility requirements.

FastEmbed declares the CPU `onnxruntime` distribution as a dependency. The installer
installs base requirements, removes conflicting ONNX Runtime distributions in the
**private GPU environment only**, then installs the selected backend. Pip may report
the CPU distribution missing there even though the replacement supplies the same
module. Do not manually reinstall base requirements inside a configured GPU
environment; use **Install / repair GPU support** instead.

The model cache uses `LOCALAPPDATA` on Windows and `XDG_DATA_HOME` (default
`~/.local/share`) on Linux, under `SubtitleSync/model-cache`. Set
`SUBTITLE_SYNC_MODEL_CACHE` to override it.

## Regression tests

Run `PYTHON -m unittest -v test_subtitle_devices test_gpu_runtime test_vulkan_embeddings test_manual_sync test_speech_to_text test_ui_preferences` using the
environment's Python (`.venv-linux/bin/python` on Linux,
`.venv\Scripts\python.exe` on Windows).
GUI tests need a display; on headless Linux use
`xvfb-run -a .venv-linux/bin/python -m unittest -v test_subtitle_devices test_gpu_runtime test_vulkan_embeddings test_manual_sync test_speech_to_text test_ui_preferences`.
The tests mock inference sessions and speech engines and do not require model
downloads or a GPU. Speech tests cover language/model validation, platform/backend
selection, audio-track mapping, timeline offsets, cancellation, output protection,
GPU-fallback rejection and the generated-reference GUI handoff.
