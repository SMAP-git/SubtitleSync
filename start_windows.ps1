param([switch] $RestoreLibraries, [switch] $RestoreComponents)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Test-Python(
    [string] $Executable,
    [string] $Code = "import sys, tkinter, venv, ensurepip, _tkinter; assert sys.version_info >= (3, 10)"
) {
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $null = & $Executable -c $Code 2>&1
        return $LASTEXITCODE -eq 0
    } catch [System.Management.Automation.CommandNotFoundException] {
        return $false
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

try {
    if (Test-Path -LiteralPath "PORTABLE") {
        $env:SUBTITLE_SYNC_SETTINGS_DIR = Join-Path $PSScriptRoot ".config"
        $env:SUBTITLE_SYNC_MODEL_CACHE = Join-Path $PSScriptRoot ".model-cache"
    }
    $venv = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venv)) {
        $python = Join-Path $PSScriptRoot ".python\windows\python\python.exe"
        if (-not ((Test-Path -LiteralPath $python) -and (Test-Python $python))) {
            $python = $null
            foreach ($candidate in @("python", "python3")) {
                $command = Get-Command $candidate -ErrorAction SilentlyContinue
                if ($command -and $command.Source -notlike "*\WindowsApps\*" -and (Test-Python $command.Source)) {
                    $python = $command.Source
                    break
                }
            }
        }
        if (-not $python) {
            if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
                throw "Automatic Python download supports Windows x64. Install Python 3.10+ with Tcl/Tk and pip for this architecture."
            }
            Write-Host "Python with Tkinter was not found."
            Write-Host "Download private CPython 3.12.15 (about 21 MB) from astral-sh/python-build-standalone into this app folder?"
            if ((Read-Host "Download? [y/N]") -notmatch "^[yY]$") {
                throw "Cancelled. You can install Python with Tcl/Tk and pip manually, then run the launcher again."
            }
            $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
            if (-not $tar) { throw "Windows tar.exe is required. Use Windows 10/11, or install Python with Tcl/Tk and pip manually." }
            $directory = Join-Path $PSScriptRoot ".python\windows"
            New-Item -ItemType Directory -Force -Path $directory | Out-Null
            $archive = Join-Path $directory "runtime.tar.gz.part"
            try {
                [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
                $url = "https://github.com/astral-sh/python-build-standalone/releases/download/20261013/cpython-3.12.15%2B20261013-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
                Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $archive
                $expected = "7a2aef48e5ee569e310b29b32681debbc5d888e92790d85331a47736c6b06aad"
                if ((Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant() -ne $expected) {
                    throw "Python download checksum mismatch. No runtime was installed."
                }
                & $tar.Source -xzf $archive -C $directory
                if ($LASTEXITCODE -ne 0) { throw "Cannot extract private Python." }
            } finally {
                if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive }
            }
            $python = Join-Path $directory "python\python.exe"
            if (-not (Test-Python $python)) {
                throw "Private Python could not run. Check the error and Windows system-library requirements."
            }
        }
        & $python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Cannot create the app environment. Extract to a writable folder." }
    }
    if (-not (Test-Python $venv)) { throw "Existing environment is not usable. Check Python/Tkinter or extract a fresh release folder." }
    Write-Host "Checking app dependencies (first launch requires internet)..."
    if (-not (Test-Python $venv "import fastembed, onnxruntime; assert fastembed.__version__ == '0.9.0'")) {
        & $venv -m pip install --disable-pip-version-check -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Check internet access and the errors above." }
    }
    if (-not (Test-Python $venv "import imageio_ffmpeg")) {
        & $venv -m pip install --disable-pip-version-check --no-deps -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "Cannot install audio support." }
    }
    if ($RestoreLibraries) {
        & $venv library_updates.py restore
        if ($LASTEXITCODE -ne 0) { throw "Could not restore previous libraries." }
    }
    if ($RestoreComponents) {
        & $venv library_updates.py restore-components
        if ($LASTEXITCODE -ne 0) { throw "Could not restore previous runtimes/models." }
    }
    $appPython = & $venv library_updates.py launch-python
    if ($LASTEXITCODE -ne 0) { throw "Cannot select app libraries. Try Start_Subtitle_Sync.bat --restore-libraries." }
    & $appPython subtitle_sync_app.pyw
    if ($LASTEXITCODE -ne 0) { throw "Subtitle Sync exited with an error." }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
