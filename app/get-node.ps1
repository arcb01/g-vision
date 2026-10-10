# Downloads the current Node.js LTS as a portable folder into -Dest: no
# installer and no admin rights, so it works where the MSI fails (exit code
# 1603, often a leftover or half-removed Node.js install). prereqs.cmd calls
# it when no Node.js 22+ is found.
param([Parameter(Mandatory)][string]$Dest)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # the progress bar makes downloads crawl
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'x64' }
$lts = Invoke-RestMethod 'https://nodejs.org/dist/index.json' | Where-Object { $_.lts } | Select-Object -First 1
$version = $lts.version
$name = "node-$version-win-$arch"
$base = "https://nodejs.org/dist/$version"
Write-Host "Downloading Node.js $version ($arch)..."

# Unzip next to -Dest so the final move stays on one drive.
$tmp = "$Dest.download"
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory $tmp -Force | Out-Null
$zip = Join-Path $tmp "$name.zip"
Invoke-WebRequest "$base/$name.zip" -OutFile $zip -UseBasicParsing

$sums = [string](Invoke-RestMethod "$base/SHASUMS256.txt")
$expected = ($sums -split "`n" | Where-Object { $_ -match "\s$name\.zip$" }) -replace '\s.*$', ''
$actual = (Get-FileHash $zip -Algorithm SHA256).Hash
if (-not $expected -or $actual -ne $expected.Trim().ToUpper()) { throw "checksum mismatch for $name.zip" }

Expand-Archive $zip $tmp
Remove-Item $Dest -Recurse -Force -ErrorAction SilentlyContinue
Move-Item (Join-Path $tmp $name) $Dest
Remove-Item $tmp -Recurse -Force
