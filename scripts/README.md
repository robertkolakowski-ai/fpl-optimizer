# scripts/

## build_team_priors.py

Leser sesongstatistikk fra `tsdl_Premier_League.xlsx` (i Scraper-prosjektet) og skriver
en kompakt `data/team_priors.json` med:

- `xg_home`, `xg_away`, `xga_home`, `xga_away` — empirisk xG/xGA per lag, hjemme/borte
- `cs_pct_home`, `cs_pct_away`, `cs_pct_overall` — clean-sheet-rate
- `goals_first_half_pct`, `goals_second_half_pct` — andel mål 1./2. omgang
- `btts_pct_overall` — both teams to score-rate
- Avledet: `xg_avg`, `xga_avg`, `net_xg`, `home_advantage_xg`

Kilden finnes via:
1. `$env:TSDL_PL_XLSX` — full sti
2. `../Scraper/output/tsdl_Premier_League.xlsx` (relativt repo-rot)

Kjør manuelt:
```bash
python scripts/build_team_priors.py
```

## update_priors.ps1

Kombinerer `build_team_priors.py` med git commit + push. Skal kjøres 2× per uke.

### Sett opp Task Scheduler (én gang)

Åpne en PowerShell som **Administrator**:

```powershell
schtasks /create `
  /tn "FPL Update Priors" `
  /tr "powershell.exe -ExecutionPolicy Bypass -File 'C:\Users\rober\Claude prosjekter\fpl-optimizer\scripts\update_priors.ps1'" `
  /sc weekly `
  /d MON,THU `
  /st 06:30 `
  /rl HIGHEST
```

Det oppretter en task som kjører hver mandag og torsdag kl. 06:30.

### Test scriptet manuelt

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\update_priors.ps1"
```

Logger til `data/update_priors.log`.

### Hva skjer ved endring

1. Konverter-en kjører og skriver ny `data/team_priors.json`
2. Hvis git ser endringer i den filen → commit + push til master
3. Render auto-deployer (~2-3 min senere)
4. Backend-modulen `team_priors.py` poller filens mtime og laster inn på nytt
   ved første request etter deploy — ingen restart trengs

### Hvis Scraper-filen ikke er oppdatert

Scriptet exitter raskt uten å committe. Trygt å kjøre selv om kilden er stale.

### Slett task

```powershell
schtasks /delete /tn "FPL Update Priors" /f
```
