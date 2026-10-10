# Sets up and starts G-VISION on Windows. G-VISION.bat runs it on every
# launch. Anything missing (Node.js, Git, Python) is downloaded as a
# portable copy into runtime\, so a fresh PC needs no installers and no admin
# rights. A copy downloaded as a ZIP is turned into a git checkout so it can
# update itself. Then app\update.js pulls the latest version and installs
# dependencies when they changed, Electron is downloaded, and the app opens.
# Messages also go to logs\setup.log.
#
# Environment knobs, used by CI:
#   GVISION_SETUP_ONLY=1     set everything up but do not open the app
#   GVISION_REPO, GVISION_BRANCH   where a ZIP copy gets its git history from
#   GVISION_EXTRAS, GVISION_NO_GPU are read by app\update.js
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # the progress bar makes downloads crawl
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$AppDir = $PSScriptRoot
$Root = Split-Path $AppDir
$Runtime = Join-Path $Root 'runtime'
$Logs = Join-Path $Root 'logs'
$Repo = if ($env:GVISION_REPO) { $env:GVISION_REPO } else { 'https://github.com/arcb01/g-vision.git' }
$Branch = if ($env:GVISION_BRANCH) { $env:GVISION_BRANCH } else { 'main' }
$Arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'x64' }

New-Item -ItemType Directory $Logs -Force | Out-Null
try { Start-Transcript (Join-Path $Logs 'setup.log') | Out-Null } catch {}

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Add-Path($dir) { $env:Path = "$dir;$env:Path" }
function Has($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

# Runs a program, showing its output, and throws if it fails.
# No named parameter, so flags like -C or -e pass straight through.
function Run {
  $exe, $rest = $args
  & $exe @rest
  if ($LASTEXITCODE -ne 0) { throw "$($args -join ' ') failed (exit code $LASTEXITCODE)" }
}

function Get-Json($url) {
  $headers = @{ 'User-Agent' = 'g-vision-setup' }
  if ($env:GITHUB_TOKEN -and $url -like 'https://api.github.com/*') { $headers.Authorization = "Bearer $env:GITHUB_TOKEN" }
  Invoke-RestMethod $url -Headers $headers -UseBasicParsing
}

# Downloads $url to $out and checks its SHA-256.
function Download($url, $out, $sha256) {
  Write-Host "Downloading $url"
  Invoke-WebRequest $url -OutFile $out -UseBasicParsing -Headers @{ 'User-Agent' = 'g-vision-setup' }
  $actual = (Get-FileHash $out -Algorithm SHA256).Hash
  if (-not $sha256 -or $actual -ne $sha256.Trim().ToUpper()) { throw "checksum mismatch for $url" }
}

# Unzips $zip into $dest, replacing it. $inner is the zip's top folder, if any.
function Expand-To($zip, $dest, $inner) {
  $tmp = "$dest.unzip"
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
  Expand-Archive $zip $tmp
  Remove-Item $dest -Recurse -Force -ErrorAction SilentlyContinue
  $src = if ($inner) { Join-Path $tmp $inner } else { $tmp }
  Move-Item $src $dest
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item $zip -Force
}

function Test-Node {
  if (-not (Has node)) { return $false }
  try { $v = & node -p 'process.versions.node' } catch { return $false }
  return $LASTEXITCODE -eq 0 -and [int]("$v".Split('.')[0]) -ge 22
}

function Install-Node {
  $lts = Get-Json 'https://nodejs.org/dist/index.json' | Where-Object { $_.lts } | Select-Object -First 1
  $name = "node-$($lts.version)-win-$Arch"
  $base = "https://nodejs.org/dist/$($lts.version)"
  $sums = [string](Invoke-RestMethod "$base/SHASUMS256.txt" -UseBasicParsing)
  $line = $sums -split "`n" | Where-Object { $_ -match "\s$([regex]::Escape($name)).zip$" }
  New-Item -ItemType Directory $Runtime -Force | Out-Null
  $zip = Join-Path $Runtime "$name.zip"
  Download "$base/$name.zip" $zip ("$line" -split '\s+')[0]
  Expand-To $zip (Join-Path $Runtime 'node') $name
}

function Install-Git {
  $rel = Get-Json 'https://api.github.com/repos/git-for-windows/git/releases/latest'
  $bits = if ($Arch -eq 'arm64') { 'arm64' } else { '64-bit' }
  $asset = $rel.assets | Where-Object { $_.name -match "^MinGit-[\d.]+-$bits\.zip$" } | Select-Object -First 1
  if (-not $asset) { throw "no MinGit download in git-for-windows $($rel.tag_name)" }
  New-Item -ItemType Directory $Runtime -Force | Out-Null
  $zip = Join-Path $Runtime $asset.name
  Download $asset.browser_download_url $zip ("$($asset.digest)" -replace '^sha256:', '')
  Expand-To $zip (Join-Path $Runtime 'git') $null
}

function Install-Uv {
  $name = "uv-$(if ($Arch -eq 'arm64') { 'aarch64' } else { 'x86_64' })-pc-windows-msvc.zip"
  $base = 'https://github.com/astral-sh/uv/releases/latest/download'
  New-Item -ItemType Directory $Runtime -Force | Out-Null
  $sha = (([string](Invoke-RestMethod "$base/$name.sha256" -UseBasicParsing)).Trim() -split '\s+')[0]
  $zip = Join-Path $Runtime $name
  Download "$base/$name" $zip $sha
  Expand-To $zip (Join-Path $Runtime 'uv') $null
}

try {
  Write-Host "G-VISION setup in $Root"

  # Portable copies first so they win over an older system install.
  Add-Path (Join-Path $Runtime 'node')
  Add-Path (Join-Path $Runtime 'git\cmd')
  Add-Path (Join-Path $Runtime 'uv')

  if (-not (Test-Node)) {
    Step 'Node.js 22 or newer not found: downloading a portable copy into runtime\node'
    Install-Node
    if (-not (Test-Node)) { throw 'the downloaded Node.js does not run' }
  }

  # Git on a drive formatted FAT/exFAT or owned by another user refuses to
  # work ("dubious ownership"); this applies to this window and its children.
  $env:GIT_CONFIG_COUNT = '1'
  $env:GIT_CONFIG_KEY_0 = 'safe.directory'
  $env:GIT_CONFIG_VALUE_0 = '*'

  if (-not (Has git)) {
    Step 'Git not found: downloading a portable copy into runtime\git (for updates)'
    try { Install-Git } catch { Write-Warning "Could not get Git, so G-VISION cannot update itself: $_" }
  }
  if ((Has git) -and -not (Test-Path (Join-Path $Root '.git'))) {
    # A ZIP download: attach it to the repository and move to the latest
    # version, so the next launches can update with git pull. Ignored files
    # (settings, models, logs) are kept.
    Step "Connecting this copy to $Repo for updates"
    try {
      Run git -C $Root init -q
      Run git -C $Root remote add origin $Repo
      Run git -C $Root fetch -q origin $Branch
      Run git -C $Root checkout -q -f -B $Branch "origin/$Branch"
    } catch {
      Write-Warning "Could not connect to the repository, so G-VISION cannot update itself: $_"
      Remove-Item (Join-Path $Root '.git') -Recurse -Force -ErrorAction SilentlyContinue
    }
  }

  # A Python to build python\.venv from, only needed the first time. uv
  # downloads a standalone CPython into runtime\python; update.js does the rest.
  $env:UV_PYTHON_INSTALL_DIR = Join-Path $Runtime 'python'
  if (-not (Test-Path (Join-Path $Root 'python\.venv\Scripts\python.exe'))) {
    Step 'Getting Python 3.12 into runtime\python'
    if (-not (Has uv)) { Install-Uv }
    Run uv python install 3.12
    $env:GVISION_BASE_PYTHON = (& uv python find 3.12 | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0 -or -not $env:GVISION_BASE_PYTHON) { throw 'uv could not find the Python it installed' }
  }

  Step 'Updating and installing dependencies'
  & node (Join-Path $AppDir 'update.js')
  $updateFailed = $LASTEXITCODE -eq 2

  $electron = Join-Path $AppDir 'node_modules\electron'
  if (-not (Test-Path (Join-Path $electron 'package.json'))) {
    Step "Installing the app's packages, first run only"
    Push-Location $AppDir
    try { Run npm install --no-audit --no-fund } finally { Pop-Location }
  }
  # Electron 44 and later no longer download their binary during npm install.
  # This does, or again after an update changes the version; else a no-op.
  Run node (Join-Path $electron 'install.js')
  $exe = Join-Path $electron 'dist\electron.exe'
  if (-not (Test-Path $exe)) { throw "Electron was not downloaded ($exe is missing)" }

  if ($updateFailed) {
    Write-Warning 'Something above could not be installed; G-VISION starts anyway and retries next launch.'
    Start-Sleep 20
  }
  if ($env:GVISION_SETUP_ONLY) {
    Write-Host "`nSetup finished."
  } else {
    $appArgs = @("`"$AppDir`"") + @($args | ForEach-Object { "`"$_`"" })
    Start-Process $exe -ArgumentList $appArgs -WorkingDirectory $AppDir
  }
  try { Stop-Transcript | Out-Null } catch {}
  exit 0
} catch {
  Write-Host ''
  Write-Host "G-VISION setup failed: $_" -ForegroundColor Red
  Write-Host "Check the internet connection and run G-VISION.bat again. If it keeps failing, send $Logs\setup.log."
  try { Stop-Transcript | Out-Null } catch {}
  exit 1
}
