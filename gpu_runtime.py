"""App-local GPU installation and isolated inference workers."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import threading
import venv
from pathlib import Path
from typing import Callable

APP_DIR = Path(__file__).resolve().parent
CPU_ONLY = os.environ.get("SUBTITLE_SYNC_CPU_ONLY") == "1"


class CancelledError(RuntimeError):
    pass
DRIVER_URLS = {
    "NVIDIA": "https://www.nvidia.com/Download/index.aspx",
    "AMD": "https://www.amd.com/en/support/download/drivers.html",
    "INTEL": "https://www.intel.com/content/www/us/en/download-center/home.html",
}
LINUX_URLS = {
    "NVIDIA": "https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/",
    "AMD": "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/",
    "INTEL": "https://dgpu-docs.intel.com/driver/client/overview.html",
}
MIGRAPHX_URL = "https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/"
AMD_ROCM_URL = "https://rocm.docs.amd.com/en/latest/install/rocm.html"
AMD_COMPATIBILITY_URL = "https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html"
AMD_ONNX_URL = "https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html"
AMD_SETUP_OPTIONS = ("Official AMD driver / ROCm", "MIGraphX inference")
INTEGRATED_DEVICES = {
    "AMD APU (integrated)": "AMD",
    "INTEL integrated GPU": "INTEL",
}


def device_vendor(device: str) -> str:
    return INTEGRATED_DEVICES.get(device, device)


# Suggestions are not a support matrix; users can enter an unlisted exact model.
AMD_APU_MODELS = (
    "Ryzen AI Max+ 395 (Radeon 8060S)", "Ryzen AI Max 390 (Radeon 8050S)",
    "Ryzen AI 9 HX 370 (Radeon 890M)", "Ryzen AI 9 365 (Radeon 880M)",
    "Ryzen 7 8700G (Radeon 780M)", "Ryzen 5 8600G (Radeon 760M)",
    "Ryzen 5 8500G (Radeon 740M)", "Ryzen 7 7840U (Radeon 780M)",
    "Ryzen 7 6800U (Radeon 680M)", "Ryzen 5 6600U (Radeon 660M)",
    "Ryzen 7 5700G (Radeon Vega 8)", "Ryzen 5 5600G (Radeon Vega 7)",
    "Ryzen 7 4700G (Radeon Vega 8)", "Ryzen 5 4600G (Radeon Vega 7)",
    "Ryzen 5 3400G (Radeon Vega 11)", "Ryzen 3 3200G (Radeon Vega 8)",
    "Ryzen 5 2400G (Radeon Vega 11)", "Ryzen 3 2200G (Radeon Vega 8)",
    "Athlon 3000G (Radeon Vega 3)", "A-Series APU (Radeon R7 / R5)",
)
AMD_GPU_MODELS = (
    "Radeon RX 9070 XT", "Radeon RX 9070", "Radeon RX 9060 XT",
    "Radeon RX 7900 XTX", "Radeon RX 7900 XT", "Radeon RX 7900 GRE",
    "Radeon RX 7800 XT", "Radeon RX 7700 XT", "Radeon RX 7600 XT", "Radeon RX 7600",
    "Radeon RX 6950 XT", "Radeon RX 6900 XT", "Radeon RX 6800 XT",
    "Radeon RX 6800", "Radeon RX 6750 XT", "Radeon RX 6700 XT", "Radeon RX 6700",
    "Radeon RX 6650 XT", "Radeon RX 6600 XT", "Radeon RX 6600",
    "Radeon RX 6500 XT", "Radeon RX 6400",
    "Radeon RX 5700 XT", "Radeon RX 5700", "Radeon RX 5600 XT", "Radeon RX 5500 XT",
    "Radeon VII", "Radeon RX Vega 64", "Radeon RX Vega 56",
    "Radeon RX 590", "Radeon RX 580", "Radeon RX 570",
    "Radeon RX 560", "Radeon RX 560 Series", "Radeon RX 550",
    "Radeon RX 480", "Radeon RX 470", "Radeon RX 460",
    "Radeon R9 Fury X", "Radeon R9 Fury", "Radeon R9 390X", "Radeon R9 390",
    "Radeon R9 380", "Radeon R9 290", "Radeon R7 370", "Radeon R7 360",
    "Radeon PRO W7900", "Radeon PRO W7800", "Radeon PRO W6800",
    "Radeon AI PRO R9700",
    "Instinct MI350X", "Instinct MI325X", "Instinct MI300X",
    "Instinct MI300A", "Instinct MI250X", "Instinct MI210", "Instinct MI100",
) + AMD_APU_MODELS
AMD_DISTRIBUTIONS = (
    "Ubuntu", "Debian", "Red Hat Enterprise Linux", "Rocky Linux",
    "Oracle Linux", "SUSE Linux Enterprise Server", "Fedora",
    "Arch Linux", "Linux Mint", "openSUSE",
)


def linux_distribution() -> tuple[str, str]:
    release = platform.freedesktop_os_release()
    names = {
        "ubuntu": "Ubuntu", "debian": "Debian", "rhel": "Red Hat Enterprise Linux",
        "rocky": "Rocky Linux", "ol": "Oracle Linux",
        "sles": "SUSE Linux Enterprise Server", "fedora": "Fedora",
        "arch": "Arch Linux", "linuxmint": "Linux Mint", "opensuse-leap": "openSUSE",
        "opensuse-tumbleweed": "openSUSE",
    }
    return names.get(release.get("ID", ""), release.get("NAME", "")), release.get("VERSION_ID", "")


def amd_linux_guidance(setup: str, gpu: str, distribution: str, version: str) -> tuple[str, str]:
    if setup not in AMD_SETUP_OPTIONS:
        raise ValueError("Choose Official AMD driver / ROCm or MIGraphX inference.")
    if not gpu.strip() or not distribution.strip() or not version.strip():
        raise ValueError("Select or enter your exact GPU model, Linux distribution and release.")
    gpu, distribution, version = gpu.strip(), distribution.strip(), version.strip()
    text = (
        f"Selected: {gpu}; {distribution} {version}.\n"
        "Compatibility is NOT verified. Model suggestions are not a list of supported "
        "GPUs. Check this exact GPU, OS release, kernel and ROCm version in AMD's "
        "current compatibility matrix before installing.\n\n"
    )
    if "rx 560" in gpu.lower() and "rx 5600" not in gpu.lower():
        text += (
            "RX 560 / Polaris: this is a legacy GPU, not a supported target in current "
            "official ROCm/MIGraphX support matrices. A working Linux graphics driver "
            "does not imply ROCm support. Select Recommended/Vulkan to use an existing "
            "Vulkan driver instead. Use CPU if neither backend works. Do not install "
            "another GPU's packages or force a different GPU architecture.\n\n"
        )
    if distribution in ("Ubuntu", "Debian", "Linux Mint"):
        text += (
            "Distribution family: apt/deb. Use instructions for your exact distribution "
            "and release; do not substitute Ubuntu packages for Debian or Mint unless AMD "
            "explicitly documents that combination.\n"
        )
    elif distribution in ("Red Hat Enterprise Linux", "Rocky Linux", "Oracle Linux", "Fedora"):
        text += (
            "Distribution family: dnf/rpm. Check the exact enterprise release and kernel; "
            "RHEL instructions do not automatically apply to Rocky, Oracle or Fedora.\n"
        )
    elif distribution in ("SUSE Linux Enterprise Server", "openSUSE"):
        text += (
            "Distribution family: zypper/rpm. SLES support does not imply openSUSE support; "
            "check the exact release in AMD's matrix.\n"
        )
    else:
        text += (
            "No distribution-specific installation path is assumed for this selection. "
            "If AMD does not list it, use CPU rather than installing another distro's packages.\n"
        )
    if "radeon" in gpu.lower() or "ryzen" in gpu.lower():
        text += "In AMD's selector, use Radeon GPU or Ryzen APU, as appropriate.\n"
    elif "instinct" in gpu.lower():
        text += "In AMD's selector, use Instinct GPU.\n"
    else:
        text += "Confirm the exact AMD GPU family and model in the official selector.\n"
    text += (
        "The guide opens AMD's own selector; repeat your GPU/OS selection there. "
        "The app does not preselect website fields or guarantee an available package.\n\n"
    )
    if setup == AMD_SETUP_OPTIONS[0]:
        return text + (
            "Official AMD driver / ROCm setup:\n"
            "1. Open the compatibility matrix, then the selected setup guide.\n"
            "2. Choose your exact GPU and OS release on AMD's website and follow its "
            "driver/ROCm instructions. System changes may require root and a reboot.\n"
            "3. Return here and select MIGraphX inference. Driver/ROCm installation "
            "alone does not supply this app's ONNX Runtime inference backend.\n"
            "No system commands are run by this app."
        ), AMD_ROCM_URL
    directory = runtime_directory("AMD")
    python = shlex.quote(str(runtime_python("AMD")))
    return text + (
        "MIGraphX inference setup (requires compatible AMD driver/ROCm first):\n"
        "1. Open the selected setup guide for MIGraphX and the ONNX Runtime guide "
        "for the matching MIGraphXExecutionProvider wheel/build. The MIGraphX Python "
        "package alone is not enough.\n"
        "2. Use a Python version supported by AMD's build to create this environment:\n"
        f"python3 -m venv {shlex.quote(str(directory))}\n"
        f"{python} -m pip install -r {shlex.quote(str(APP_DIR / 'requirements.txt'))}\n"
        f"{python} -m pip uninstall -y onnxruntime\n"
        "3. Install the AMD ONNX Runtime wheel/build matching your ROCm, Python, "
        "architecture and OS into that environment using its Python. Do not install "
        "the NVIDIA onnxruntime-gpu package. No wheel is chosen automatically.\n"
        "4. Click Test GPU. Only a successful inference test registers the environment. "
        "Your model selection does not change the GPU adapter ID; choose that on the main screen."
    ), MIGRAPHX_URL


def base_runtime_directory(device: str) -> Path:
    if CPU_ONLY:
        raise ValueError("GPU support is disabled in the CPU-only image. Use the full GPU image.")
    device = device_vendor(device)
    if device not in DRIVER_URLS:
        raise ValueError(f"Unknown GPU vendor: {device}.")
    if sys.platform != "win32" and not sys.platform.startswith("linux"):
        raise ValueError("GPU setup supports Windows and Linux only.")
    backend = "directml" if sys.platform == "win32" and device != "NVIDIA" else device.lower()
    platform = "windows" if sys.platform == "win32" else "linux"
    return APP_DIR / ".gpu-runtimes" / f"{platform}-{backend}"


def runtime_directory(device: str) -> Path:
    from managed_updates import selected_path
    base = base_runtime_directory(device)
    return selected_path(f"gpu-{base.name}", base)


def runtime_python(device: str) -> Path:
    directory = runtime_directory(device)
    return directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def runtime_ready(device: str) -> bool:
    return runtime_python(device).is_file() and (runtime_directory(device) / "installed").is_file()


def runtime_package(device: str) -> str:
    device = device_vendor(device)
    runtime_directory(device)
    if device == "NVIDIA":
        # 1.26.0 is the newest CUDA 12 build; it requires Python 3.11+.
        version = "1.26.0" if sys.version_info >= (3, 11) else "1.23.2"
        return f"onnxruntime-gpu[cuda,cudnn]=={version}"
    if sys.platform == "win32":
        return "onnxruntime-directml"
    if device == "INTEL":
        return "onnxruntime-openvino"
    raise ValueError(
        "AMD on Linux requires a ROCm/MIGraphX build matched to your GPU and "
        "distribution. Open the AMD Linux and MIGraphX guides in GPU setup. "
        f"Install it in {runtime_directory(device)}, then use Test GPU."
    )


def run_command(command: list[str], progress: Callable[[str], None]) -> None:
    progress(" ".join(command))
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    ) as process:
        if process.stdout is None:
            raise RuntimeError("Could not read the installer output.")
        tail: list[str] = []
        for line in process.stdout:
            line = line.rstrip()
            if line:
                progress(line)
                tail = (tail + [line])[-20:]
        if process.wait() != 0:
            raise ValueError(
                "GPU runtime installation failed. Your Python version, platform or "
                "network may not support this package. CPU mode is unchanged.\n\n"
                + "\n".join(tail)
            )


def install_runtime(device: str, progress: Callable[[str], None]) -> None:
    package = runtime_package(device)
    directory = runtime_directory(device)
    if directory != base_runtime_directory(device):
        raise ValueError("A versioned GPU runtime is selected. Use Library updates to update or restore it.")
    marker = directory / "installed"
    marker.unlink(missing_ok=True)
    progress("Creating a private GPU environment (system drivers are not changed)...")
    venv.EnvBuilder(with_pip=True).create(directory)
    python = str(runtime_python(device))
    pip = [python, "-m", "pip", "--disable-pip-version-check"]
    run_command(pip + ["install", "-r", str(APP_DIR / "requirements.txt")], progress)
    run_command(
        pip + ["uninstall", "-y", "onnxruntime", "onnxruntime-gpu",
               "onnxruntime-directml", "onnxruntime-openvino"], progress,
    )
    run_command(pip + ["install", "--only-binary=:all:", package], progress)
    marker.write_text(package + "\n", encoding="utf-8")


def run_worker(
    device: str, device_id: int, request: dict[str, object],
    progress: Callable[[str], None],
    cancel: threading.Event | None = None,
) -> dict:
    python = runtime_python(device)
    if not python.is_file():
        import onnxruntime as ort
        from subtitle_alignment import execution_provider

        provider, _ = execution_provider(device, device_id)
        if provider not in ort.get_available_providers():
            raise ValueError("GPU runtime is not installed. Open GPU setup first.")
        python = Path(sys.executable)
    # Keep stdout a JSON-lines protocol; native runtime diagnostics go to stderr.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errors:
        with subprocess.Popen(
            [str(python), str(Path(__file__).resolve()), "--worker", device, str(device_id)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
            text=True, encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        ) as process:
            stopped = threading.Event()

            def monitor_cancel() -> None:
                while not stopped.wait(0.1):
                    if cancel is not None and cancel.is_set():
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                        return

            if cancel is not None:
                threading.Thread(target=monitor_cancel, daemon=True).start()
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("Could not communicate with GPU worker.")
            process.stdin.write(json.dumps(request))
            process.stdin.close()
            result = None
            failure = None
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    failure = "GPU worker returned invalid output: " + line[:500]
                    continue
                if not isinstance(event, dict):
                    failure = "GPU worker returned an invalid event."
                    continue
                if "progress" in event:
                    if cancel is None or not cancel.is_set():
                        progress(event["progress"])
                elif "error" in event:
                    failure = event["error"]
                elif "result" in event:
                    result = event["result"]
            code = process.wait()
            stopped.set()
            if cancel is not None and cancel.is_set():
                raise CancelledError("Synchronization cancelled. No output was written.")
            if code or failure or not isinstance(result, dict):
                errors.seek(0, os.SEEK_END)
                size = errors.tell()
                errors.seek(max(0, size - 12000))
                details = errors.read()
                raise ValueError(
                    f"GPU test/analysis failed: {failure or f'worker exit code {code}'}.\n"
                    "Check the adapter ID and install a compatible system driver using "
                    "GPU setup. CPU mode remains available.\n\n" + details
                )
            return result


def test_runtime(device: str, device_id: int, progress: Callable[[str], None]) -> str:
    result = run_worker(device, device_id, {"operation": "test"}, progress)
    if runtime_python(device).is_file():
        (runtime_directory(device) / "installed").write_text("tested\n", encoding="utf-8")
    return result["message"]


def worker(device: str, device_id: int) -> int:
    output = sys.stdout

    def emit(event: dict) -> None:
        output.write(json.dumps(event) + "\n")
        output.flush()

    def progress(message: str) -> None:
        emit({"progress": message})

    try:
        request = json.load(sys.stdin)
        # Dependencies can print on import; only our events belong on stdout.
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            import numpy as np
            import subtitle_alignment as alignment

            if request["operation"] == "test":
                progress("Loading model and testing GPU inference (first use may download the model)...")
                model = alignment._load_embedding_model(device, device_id)
                vectors = np.asarray(list(model.embed(["GPU test", "Teste da GPU"], batch_size=2)))
                if vectors.shape != (2, 384) or not np.isfinite(vectors).all():
                    raise ValueError("GPU inference returned invalid embeddings.")
                provider, _ = alignment.execution_provider(device, device_id)
                if provider not in model.model.get_providers():
                    raise ValueError("Selected GPU provider was lost during inference.")
                result = {"message": f"Inference passed on {provider}, adapter {device_id}."}
            elif request["operation"] == "analyze":
                anchors, diagnostics = alignment.analyze_subtitles(
                    [alignment.Cue(**cue) for cue in request["source"]],
                    [alignment.Cue(**cue) for cue in request["reference"]],
                    request["initial_audio_ms"], progress=progress,
                    hearing_impaired_strictness=request["strictness"],
                    processing_device=device, device_id=device_id,
                    source_anchor=request.get("source_anchor", 1),
                    reference_anchor=request.get("reference_anchor"),
                    automatic_initial=request.get("automatic_initial", False),
                )
                result = {"anchors": anchors, "diagnostics": diagnostics}
            else:
                raise ValueError("Unknown GPU worker operation.")
        emit({"result": result})
        return 0
    except Exception as error:
        emit({"error": str(error)})
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, metavar=("DEVICE", "ADAPTER_ID"), required=True)
    args = parser.parse_args()
    raise SystemExit(worker(args.worker[0], int(args.worker[1])))
