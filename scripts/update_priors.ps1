# update_priors.ps1
#
# Kjor 2x per uke (mandag + torsdag) via Windows Task Scheduler.
# Steg:
#   1. Bygg ferskt data/team_priors.json fra Scraper-output
#   2. Hvis filen har endret seg, git commit + push (Render auto-deployer)
#   3. Logg resultat
#
# Hvis Scraper-filen ikke er oppdatert siden sist kjoring, exit'er scriptet
# raskt uten aa committe noe.
#
# Sett opp i Task Scheduler (PowerShell som Administrator):
#   schtasks /create /tn "FPL Update Priors" /tr "powershell.exe -ExecutionPolicy Bypass -File 'C:\Users\rober\Claude prosjekter\fpl-optimizer\scripts\update_priors.ps1'" /sc weekly /d MON,THU /st 06:30 /rl HIGHEST

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogPath = Join-Path $RepoRoot "data\update_priors.log"

function Write-LogLine {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$stamp  $Message"
    $line | Out-File -FilePath $LogPath -Append -Encoding utf8
    Write-Host $line
}

Write-LogLine "=== Start update_priors ==="

try {
    Set-Location $RepoRoot

    # 1. Generer ny JSON
    Write-LogLine "Running build_team_priors.py"
    $pythonOutput = & python "scripts/build_team_priors.py"
    $joined = ($pythonOutput | Out-String).Trim()
    if ($joined) { Write-LogLine $joined }

    # 2. Sjekk om filen er endret
    $status = & git status --porcelain "data/team_priors.json"
    if (-not $status) {
        Write-LogLine "No changes to team_priors.json. Exit."
        exit 0
    }

    # 3. Commit + push
    Write-LogLine "Changes detected, committing"
    & git add "data/team_priors.json"
    $today = Get-Date -Format "yyyy-MM-dd"
    & git commit -m "Auto-update: team_priors $today"
    Write-LogLine "Pushing to origin"
    & git push origin master | Out-Null
    Write-LogLine "OK pushed. Render will redeploy automatically."
}
catch {
    $errMsg = "ERROR: " + $_.Exception.Message
    Write-LogLine $errMsg
    exit 1
}

Write-LogLine "=== End update_priors ==="
