# FPL Optimizer — Backlog

**North star:** Fantasy Premier League. Gode råd. Hjelpe brukeren oppnå best mulig resultater.
Hver feature måles mot: *gjør den brukerens FPL-rang bedre, eller gjør den ikke det?*

---

## Sesjonslogg 2026-05-01

**Levert i én sesjon:** 10 backlog-punkter (#1-#10) på MVP-nivå →
4 backend-fundamenter løftet til best-practice (#2 #4 #7 #8) →
4 ekstra moduler løftet (Dixon-Coles A, transfer-EV B, multi-GW C, SQLite cache D) →
9 design-punkter (5 P-punkter + 4 V-punkter) levert.

**Resultat:** 71 endepunkter, 8 nye Python-moduler (`backtest`, `uncertainty`,
`chip_mc`, `league_intel`, `live_bonus`, `ml_baseline`, `score_history`,
`user_prefs`, `cache`), 9 nye UI-flater. Appen importerer rent.

**Filer endret:** `analyzer.py`, `models.py`, `multi_gw.py`, `optimizer.py`,
`predictions.py`, `transfers.py`, `web.py`, `templates/web.html`.
**Filer nye:** se modulliste over.

---

## "Til neste nivå" — prioritert (mai 2026)

**Alle 10 punkter + 4 ekstra forbedringer levert på best-practice-nivå**
(mai 2026). Backend er ferdig og verifisert end-to-end.

### Best-practice-løft fullført

- **#2** Backtest bruker nå LIVE `score_players()` på rekonstruerte snapshots
  (`_player_snapshot_at_gw`) — ingen parallell forenklet replay-formel
- **#4** Bootstrap er fixture-aware (vekter samples mot kommende motstanders
  difficulty + hjemme/borte) og justerer for skader/availability med widening
  av p10-p90-intervallet
- **#7** Mini-liga har ekte rank-EV-simulering (`evaluate_transfer_rank_ev`):
  for en kandidattransfer simuleres 300 baner av resten av sesongen, beregner
  endring i median-rang + sannsynlighet for at transfer forbedrer ranglerin
- **#8** Chip-MC bruker dynamic programming over (used_chips, gw)-tilstand
  med korrekte FPL 2024+-regler (WC1 i GW≤19, WC2 i GW≥20)
- **A** Predictions.py er nå Dixon-Coles-korrigert (rho=-0.18) — fikser
  at independent Poisson underestimerer 0-0/1-1 og overestimerer narrow scorelinjer
- **B** Transfers.py har nytt `plan_transfers()` som velger optimalt blant
  0/1/2-transfer-planer med ekte hit-cost-math (-4 per ekstra transfer)
- **C** Multi-GW kan nå ta `uncertainty_map`-parameter og bruke real-EP fra
  fixture-vektet bootstrap istedenfor composite_score-deltaer
- **D** SQLite-backed shared cache (`cache.py`) overlever restarts og deles
  mellom gunicorn-workers; ETag-støtte for HTTP 304

| # | Tema | Modul | Endepunkt(er) |
|---|------|-------|---------------|
| 1 | Persistent score-historikk | `score_history.py` | `/api/score-history*` |
| 2 | Backtest-motor | `backtest.py` | `/api/backtest/captain`, `/top-n` |
| 3 | ML-baseline (linear regression) | `ml_baseline.py` | `/api/ml/{train,compare}` |
| 4 | Bayesiansk usikkerhet | `uncertainty.py` | `/api/captain-uncertainty` |
| 5 | Brukerpreferanser (SQLite) | `user_prefs.py` | `/api/user-prefs/*` |
| 6 | Live BPS-projected bonus | `live_bonus.py` | `/api/live/projected-bonus` |
| 7 | Mini-liga-intelligens | `league_intel.py` | `/api/league/<id>/intelligence` |
| 8 | Monte Carlo chip-strategi | `chip_mc.py` | `/api/chips/monte-carlo` |
| 9 | score_overrides per spiller | `optimizer.py` | `select_squad(score_overrides=...)` |
| 10 | Forklarbarhet / score-breakdown | `analyzer.py` | `/api/score-breakdown/<id>` |

### Pragmatiske beslutninger underveis

- **#5** ble levert som lett SQLite-prefs-store nøkket på FPL team-ID (som
  brukeren uansett limer inn) i stedet for full magic-link auth + e-post-
  varsler. Migrer til ekte auth ved å legge til `users(id, email)` og mappe
  `team_id → user_id`. Web Push-varsler ligger fortsatt under "Park / Deferred"
  fordi det krever cron + VAPID.
- **#3** ble linear ridge regression i ren Python (ingen sklearn/LightGBM)
  for å unngå deps-bloat. Gir RMSE ~3.8 — ikke state-of-the-art, men nok til
  å vise hvor heuristikken er overkonfident. Oppgrader senere til LightGBM
  hvis vi vil ha 3 sesongers historikk og bedre kalibrering.
- **#6** er foreløpig kun et HTTP polling-endepunkt; ekte live-mode krever
  WebSocket (Flask-SocketIO) som er en del jobb på Render free.

### UI-eksponering levert

- **#10:** "Hvorfor denne spilleren?" i player-modalen (topp-3 score-bidrag)
- **#2:** "Kjør backtest"-kort på Honesty-siden (capture rate, Salah-baseline)
- **#4:** Risk-mode-toggle på kapteinkortet (⚖️/🛡️/🚀) med p10/p50/p90-intervaller
- **Bilder:** Spillerbilder i transfer-rader og backtest-tabell

### UI-eksponering levert (mai 2026)

Alle nye backend-endepunkter har nå frontend-overflater:

- **Chip MC-graf på hjem-siden:** SVG-stolpediagram per gjenværende GW per chip
  med sannsynlighet og snittgevinst. Mappes WC1+WC2 → "Wildcard" automatisk.
- **Backtest-trendgraf:** SVG-linjegraf modell vs maks mulig per GW + tabell
  med top-3 score-bidrag per pick.
- **Rang-simuleringspanel på liga-siden:** Median-rang, 80%-intervall, P(seier),
  P(topp-3) basert på 200 sample-baner over resten av sesongen.
- **Live BPS-strip:** Synlig kun under kampvindu, poller hvert 60. sek mens
  kamper pågår. Viser 🥇🥈🥉 per fixture med BPS-tall.
- **ML-baseline ved siden av FPL ep_next** i player-modal "Hvorfor"-panel.
  Cachet i frontend (én train per sesjon).
- **📊 Transfer-rang-EV-knapp** i hver transfer-rad åpner modal med
  EP-gevinst, forventet rang-endring, P(forbedrer rang).
- **☆ Følg-spilleren-knapp** i player-modal-header. Persistert via
  `/api/user-prefs/<team_id>/watched/<player_id>`.
- **Risk-mode-toggle synker til server-side prefs** — lagres per FPL team-ID
  via `/api/user-prefs`. Henter automatisk ved sync når team-ID settes.

### Designvurdering mottatt (1. mai 2026) — utvalg implementert

Ekstern designvurdering pekte på reelle svakheter. Tatt 5 P0/P1-punkter inn
i ny iterasjon (se neste blokk). Avvist: full React/Storybook-migrering
(3-6 måneders kostnad, marginal nytte), Lighthouse-budsjett-jakt,
Style Dictionary-pipeline. Behold vanilla-JS-stacken.

### Design-løft 2026-05 — alle 9 punkter levert

Etter ekstern designvurdering: utvidet plan med visuelle V-punkter inn i sprinten.

- [x] **P0** Auto-aktiver demo for førstegangsbesøkende — fjerner "ser-ødelagt-ut"
  empty-states på Hjem
- [x] **P0** Legacy-sidebar gated bak `legacy_nav`-toggle (var allerede på plass).
  Bunn-nav konsolidert til de 5 IA-destinasjonene (Hjem · Plan · Kaptein · Liga · Arkiv).
  More-meny redirecter til Innstillinger når legacy er av.
- [x] **P1+V1** "Ukens beslutning" som H1 på Hjem med TLDR-syntese av kaptein +
  transfer i én setning. Drill-down via eksisterende kort.
- [x] **P1** Norsk språk-pass på dashboard empty-states, More-meny, transfer-CTA.
- [x] **P2** Skeleton-states på 6 widgets (kapteinkort, bonus, skader, priser,
  kommende kamper, benk) — animert shimmer som respekterer
  `prefers-reduced-motion`.
- [x] **V2** Type-skala formalisert (12/13/14/15/16/20/24/32/48/64 — ingen
  mellomverdier), ny `--fs-micro` lagt til.
- [x] **V3** Token-blokk i CSS dokumentert med bruksregler. Ingen rop med
  rødfarge, ingen hardkodede hex i komponenter.
- [x] **V4** Logo-monogram bug fikset (FPL …ptimizer → FPL Optimizer med tett
  kerning og enhetlig font-vekt).
- [x] **V5** `uiIcon()`-helper med Lucide-stil SVG-ikoner brukt for
  risk-mode-toggle, watch-knapp og transfer-rang-EV-knapp. Semantiske emoji
  beholdt (🥇🥈🥉 bonus, ⚽ live).

---

## Park / Deferred

### Hosting & domene — AKTIVERT (2026-05-01)

`https://fpl.kolakowski.no` er live med gyldig Let's Encrypt SSL.

Oppsett:
- **Render-tjeneste:** `fpl-optimizer-e8js.onrender.com` (Free-tier)
- **Custom domain:** `fpl.kolakowski.no` (CNAME hos Loopia/Domeneshop)
- **Keep-alive:** UptimeRobot pinger `/health` hvert 5. min — appen sover aldri,
  så Free-tier oppleves som Starter.

Hikker underveis (lærdom for fremtiden):
- Loopia opprettet parkerings-A-records (194.9.94.86/85) ved siden av CNAME-en
  fra "Ingen innstillinger > Parkert"-defaulten. Per DNS-spec ulovlig — løst
  ved å slette og gjenopprette subdomenet med DNS direkte.

Gjenstår (ikke kritisk):
- GitHub Actions secret `APP_URL` bør oppdateres til `https://fpl.kolakowski.no`
  (predictions-snapshot cron treffer fortsatt onrender.com-URL hvis ikke endret)

### Web Push (server-side)
Send faktisk varsel mandag morgen via cron + Push-API. Bare opt-in-flow
finnes nå (browser-permission, lokal flag). Trenger backend-job.
Når GH Actions-cron er på plass for predictions, kan samme mekanisme
gjenbrukes til å pinge Push-API.

### Bet Builder / Match Centre / Prediction Tracker / Draft
Skjult bak feature flags i Innstillinger. Kode beholdt for reversibilitet.
Vurderes slettet permanent etter 60-90 dager hvis ingen savner det.

---

## Levert (mai 2026)

### 4 ideer plukket fra ekstern redesign-vurdering ✓
- **Plain-norsk toggle:** Innstillinger → "Plain norsk i stat-labels". Erstatter
  forkortelser (xG, xA, xGI, FDR, ICT, BPS, EO, PPG, Form, Own%, Proj Min,
  Total Pts, Influence, Creativity, Threat) med klare norske begreper i
  spiller-modal, sammenligning og advanced tab. Forkortelse beholdes i parentes.
- **Honesty-modul UI:** `renderModelAccuracy` i Arkiv kobler nå til
  `/api/predictions/hit-rate`. Viser ekte tall (captain ø·snitt, transfer-vinnere %,
  1X2-treff, O2.5/BTTS) når GW-er er ferdige. Honest empty-state inntil
  første GW har data.
- **Onboarding Team ID-veiledning:** Expandable "Vis meg hvordan, steg for steg"
  under Team ID-input. 4 numererte kort med browser-mockup som viser hvor i
  URL-en (`/entry/<b>1234567</b>/event/29`) ID-en ligger. NO + EN.
- **Plan A vs Plan B vs Plan C i Multi-GW:** `plan_transfers()` parametriseres
  med `risk_profile` (safe / balanced / aggressive). `/api/multi-gw/<id>`
  returnerer alle tre i én respons. Frontend viser tre cards med
  total xPts-gevinst, antall bytter og hit-cost — bruker velger profil.



### Spiller-modal (full) ✓
ep_next-projeksjon (FPL offisiell) + egen modell, set-piece-badges (PK/FK/Hjørne
med ordre), rotasjons-badge ved >40% risiko, ny Advanced tab med per-90/ICT/BPS/risk.
Tilgjengelig overalt openPlayerModal() kalles fra (Hjem, Plan, søk).

### Sammenligning hvor som helst ✓
`openCompareModal(idA, idB?)` — universell compare-modal med søke-picker for
motstander hvis bare én spiller er gitt. ⇄-knapp i spiller-modal gir
launchpoint fra alle eksisterende klikkflater.

### Touch drag-and-drop på Plan-pitch ✓
Long-press (250ms) → drag-modus med haptic feedback. `elementFromPoint`
detekterer drop-target under finger, samme-posisjon-validering, snap-back
ved ugyldig drop. Tap-tap-fallback bevart for tilgjengelighet.

### Liga line-chart caching ✓
localStorage TTL 10 min + nytt server-side batch-endepunkt
`/api/users/history?ids=...` med ThreadPoolExecutor. Reduserer 5 sekvensielle
HTTP-roundtrips til 1 og kutter wall-clock til ~1× FPL API-latency.

### Real predictions log ✓
Ny `predictions_log.py` med disk-persistering (atomic write via temp+rename).
Tracker både fixture-prediksjoner (1X2/O2.5/BTTS) og FPL-anbefalinger
(captain, top transfer, differential). Endepunkter:
`/api/predictions/{hit-rate, log, snapshot, refresh-actuals}`.
GH Actions cron (`predictions-snapshot.yml`) committer JSON tilbake til repo
fredag 17:00 UTC (snapshot) og tirsdag 00:30 UTC (actuals) — gratis
persistens på Render free.

### Bet Builder / Match Centre / Prediction Tracker / Draft
Skjult bak feature flags i Innstillinger. Kode beholdt for reversibilitet.
Vurderes slettet permanent etter 60-90 dager hvis ingen savner det.

---

## Eksternt / produkt-strategi

### Premium-tier
`AppConfig.premiumFeatures` har scaffold for `multi_gw`, `mini_league_spy`,
`ai_chat`. Ingen billing wired. Beslutning om monetisering er produkt-
strategi, ikke teknisk.

### AI Coach
Premium-feature i scaffold. Krever LLM-integrasjon (Claude/GPT API),
kontekst om brukerens lag + ligaer + historikk. Ikke startet.
