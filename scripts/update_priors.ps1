# update_priors.ps1
#
# Kjør 2× per uke (mandag + torsdag) via Windows Task Scheduler.
# Steg:
#   1. Bygg ferskt data/team_priors.json fra Scraper-output
#   2. Hvis filen har endret seg — git commit + push (Render auto-deployer)
#   3. Logg resultat
#
# Hvis Scraper-filen ikke er oppdatert siden sist kjøring, gjør scriptet
# ingenting (rask exit).
#
# Sett opp i Task Scheduler:
#   schtasks /create /tn "FPL Update Priors" /tr "powershell -File 'C:\Users\rober\Claude prosjekter\fpl-optimizer\scripts\update_priors.ps1'" /sc weekly /d MON,THU /st 06:30
#

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogPath = Join-Path $RepoRoot "data\update_priors.log"

function Log-Line {
    param([string]$msg)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$stamp  $msg" | Out-File -FilePath $LogPath -Append -Encoding utf8
    Write-Host "$stamp  $msg"
}

Log-Line "=== Start update_priors ==="

try {
    Set-Location $RepoRoot

    # 1. Generer ny JSON
    Log-Line "Running build_team_priors.py ..."
    $pythonOutput = python "scripts/build_team_priors.py" 2>&1
    Log-Line ($pythonOutput | Out-String).Trim()

    # 2. Sjekk om filen er endret
    $status = git status --porcelain "data/team_priors.json"
    if (-not $status) {
        Log-Line "No changes to team_priors.json — exit."
        exit 0
    }

    # 3. Commit + push
    Log-Line "Changes detected, committing ..."
    git add "data/team_priors.json"
    $today = Get-Date -Format "yyyy-MM-dd"
    git commit -m "Auto-update: team_priors $today"
    Log-Line "Pushing to origin ..."
    git push origin master 2>&1 | Out-Null
    Log-Line "OK — pushed. Render will redeploy automatically."

} catch {
    Log-Line ("ERROR: " + $_.Exception.Message)
    exit 1
}

Log-Line "=== End update_priors ==="
