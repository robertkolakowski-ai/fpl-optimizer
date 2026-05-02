# FPL Optimizer — Backlog

**North star:** Fantasy Premier League. Gode råd. Hjelpe brukeren oppnå best mulig resultater.
Hver feature måles mot: *gjør den brukerens FPL-rang bedre, eller gjør den ikke det?*

---

## Gjenstående oppgaver (per 2026-05-02)

Funksjonaliteten er ferdig utviklet — disse punktene er **aktivering og drift**.

### A · Aktiver push-varsler (krever VAPID-keys)

Push-stack (`push_notifications.py` + service worker) er bygget og deployet,
men trenger VAPID-nøkler for å faktisk sende varsler.

**Steg:**

1. Generer VAPID-nøkkelpar lokalt:
   ```bash
   pip install py-vapid
   vapid --gen
   # Eller online: https://web-push-codelab.glitch.me/
   ```
   Output gir public/private key.

2. I [Render Dashboard](https://dashboard.render.com) → fpl-optimizer →
   Environment, legg til:
   - `VAPID_PUBLIC_KEY` — fra steg 1
   - `VAPID_PRIVATE_KEY` — fra steg 1
   - `VAPID_SUBJECT` = `mailto:robert@kolakowski.no`
   - `CRON_TOKEN` — random 32-tegns string (beskytter `/api/push/dispatch`)

3. Render restartet automatisk. Toggle "🔔 Aktiver varsler" i
   Innstillinger virker da på `https://fpl.kolakowski.no`.

4. Test manuelt: `curl -X POST https://fpl.kolakowski.no/api/push/test/2006459`
   etter du har subscriba.

**Status:** ikke aktivert. Estimert tid: 15 min.

### B · Sett opp cron-jobber for varsler

Når VAPID er aktivert, må noe trigger `/api/push/dispatch` for å sende
ekte varsler. Tre cron-jobber bør settes opp:

| Trigger | Schedule | Endepunkt-payload |
|---------|----------|-------------------|
| **Deadline-påminnelse** | Fredag 09:00 (24t før kickoff) | `{"kind": "deadline", "team_id": <id>}` |
| **Skadekontroll** | Hver time | `{"kind": "injury", "team_id": <id>, "body": "Saka flagget — sjekk laget"}` |
| **Prisendringer** | Daglig 02:00 | `{"kind": "price", "team_id": <id>, "body": "Cherki +0.1M"}` |

**Implementasjons-alternativer:**

- **GitHub Actions** (anbefalt — gratis, allerede etablert mønster):
  Ny `.github/workflows/push-notifications.yml` som ligner
  `predictions-snapshot.yml`. Bruker `APP_URL` + `CRON_TOKEN` som secrets.
  Itererer over alle team-IDs i `data/user_prefs.db` (eller bare ditt eget
  i starten).
- **Render Cron Job** ($1/mnd per cron) — Render har egen cron-tjeneste
  som er mer robust enn GH Actions for tidsfølsomme trigger.

**Status:** ikke utviklet. Estimert tid: 1 dag for å bygge robuste varsler
med riktig logikk (når er deadline reelt, hvordan detect skader, etc.).

### C · End-to-end-test på mobil + desktop

Etter Render-deploy av siste commit, gjør en grundig testrunde:

- [ ] Hard refresh `https://fpl.kolakowski.no` på desktop (Ctrl+Shift+R)
- [ ] Hard refresh på mobil (eller installer som PWA)
- [ ] Gå gjennom alle 5 IA-destinasjoner: Hjem · Plan · Kaptein · Liga · Lag · Arkiv
- [ ] Test mørk modus toggle (Innstillinger > Utseende)
- [ ] Test PWA-install (Chrome: ⋮ → "Install app")
- [ ] Klikk på en spiller — sjekk at lag-DNA + foto + ML-baseline vises
- [ ] Klikk Liga → Chelsea → sjekk at rang-simulering med narrativ vises
- [ ] Klikk Lag → bytt mellom Premier League / Bundesliga
- [ ] Klikk Spiller → sjekk Differensial-radar (4 modus)
- [ ] Klikk del-knapp på kapteinkort — verifiser tekst-format
- [ ] Compare 4 spillere (åpne en spiller → Sammenlign → Legg til)
- [ ] Test fixture-tooltip (hover på kamp i top-bar)

**Status:** krever menneskelig testing. Estimert tid: 30 min.

### D · Feilsøk og polerings basert på testing

Etter punkt C vil vi finne ting som ikke virker som forventet. Eksempler
fra siste sesjoner: roterende tekst, tomrom mellom sidebar og innhold,
£0.0 i kort, etc. Disse fixes inkrementelt.

**Status:** alltid pågående. Tidsbruk: avhengig av funn.

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

### Sesjonslogg 2026-05-02 (sen kveld) — 10 best-in-class-løft

Komplett implementasjon av "10 viktigste løft for å nå best-in-class".
Alle 10 levert med best-practice-tilnærming (UX, visuell design, kvalitativ
informasjon).

**Visuell foundation (#5, #4, #6):**

- **#5 Lag-farger + foto:** TEAM_COLORS-mapping (keyed by FPL team code)
  med [primary, on-primary, secondary] per Premier League-klubb. Helpere
  `teamColors(teamId)`, `teamStripe()`, `teamPill()`. Plan-pitchen har nå
  foto + lag-farget posisjons-stripe på toppen av hver tile. Player-modal-
  header med 3px venstre-stripe + foto-ramme i lagets primærfarge.
- **#4 Animasjoner:** CSS-tokens for ease-funksjoner og varigheter.
  page-fade ved nav-endring (240ms ease-out + 6px translateY), hover-lift
  på interaktive kort, knappe-press scale 0.97, captain-armbånd puls (2.4s
  loop). `animateNumber()`-helper teller opp/ned med ease-out-cubic over
  700ms, flash-fargen grønn/rød underveis. `prefers-reduced-motion`
  respektert globalt.
- **#6 Mørk modus:** Token-overrides under `html[data-theme="dark"]` med
  Ink-Slate-palett, AA-kontrast på all tekst. Auto-respekt for
  `prefers-color-scheme: dark`. FOUC-prevention via inline-script i
  `<head>`. Toggle i Innstillinger > Utseende cycler System / Mørk / Lys,
  lagrer i localStorage.

**Plattform (#3):**

- **#3 PWA installability:** `manifest.webmanifest` med navn, theme-color
  amber, three icon-størrelser (192, 512, maskable-512). SVG-ikoner med
  logo-monogrammet (sirkel + femkant). Service worker (`/sw.js`) med tre
  strategier: stale-while-revalidate for statiske, network-first for
  `/api/*` med offline-fallback, passthrough for `/health` og predictions.
  Apple touch-icon + meta-tags for iOS "Add to Home Screen".

**Onboarding & innholdsdypde (#9, #8, #10):**

- **#9 Onboarding-tour:** Erstatter gammel 11-stegs engelsk tour med 7
  norske steg ankret i de 5 IA-destinasjonene + Lag-DNA. Auto-trigget
  1.5s etter første `loadUser()`-success (markert i localStorage).
  Tour-tooltip-knapper oversatt basert på currentLocale.
- **#8 Differensial-radar:** Øverst på Spiller-siden, før søk. 4 filter-
  modus: Differensial (form>=5, eierskap<10%, lette kamper), Toppform
  (form>=6), Verdi (poeng-per-million), Stigende pris. Topp 10 per modus
  med foto, navn, lag, form, eierskap, pris, total-poeng. Lag-farget
  venstre-stripe per rad.
- **#10 Sharing/eksport:** `shareOrCopy()`-helper med Web Share API
  (mobil) → clipboard (desktop) → prompt-fallback. `showToast()` for
  bekreftelser. `shareCaptain()` genererer kuratert tekst med kapteinpick
  + meta + 'fordi'-tekst + visekaptein. Del-knapp øverst-høyre på
  hjem-kapteinkortet.

**Backend-tunge funksjoner (#1, #2):**

- **#1 Push-varsler (Web Push via VAPID):** `push_notifications.py` med
  SQLite-store for subscriptions. 5 nye API-endepunkter:
  `/api/push/{public-key, subscribe, unsubscribe, test, dispatch}`. SW
  har `push`-event handler + `notificationclick` for å åpne URL.
  Innstillinger har "🔔 Aktiver varsler"-toggle med detection av
  ikke-støtte / ikke-konfigurert. Tre planlagte varsel-typer (deadline,
  injury, price). **Aktivering krever VAPID_PUBLIC_KEY,
  VAPID_PRIVATE_KEY, VAPID_SUBJECT, CRON_TOKEN i Render env-vars.**
  pywebpush + openpyxl lagt til requirements.
- **#2 Live match-day:** Mini-liga battle på Hjem (over BPS-strippen).
  Synlig kun under match-day, auto-poll hver 30. sek. Viser dine GW-
  poeng + 4 toppe rivaler med live delta ('Per +12 leder' grønnt,
  '-7 taper' rødt). Animert rød live-prikk øverst.

**Power-tool (#7):**

- **#7 4-spiller compare:** `CompareState` refaktorert fra `{aId, bId}`
  til `{ids: [...]}`. `openCompareModal(...args)` tar opptil 4 ids
  variadic. Tabell-layout med 1 stat-rad per metrikk × N kolonner.
  Vinner-verdi per rad fargelagt grønn. Add-picker med "Legg til
  spiller (N/4)". Horizontal scroll på små skjermer.

### Sesjonslogg 2026-05-02 (kveld) — Lag-DNA + forklarbar-DNA-løft

**Forklarbar-DNA på 7 flater** — alle bruker form/kamper/eierskap/risiko-
vokabular i klartekst og leder til konkret handling:
- Liga rang-simulering: narrativ analyse (median, 80%-intervall, P(seier),
  konkret handling basert på posisjon)
- Chip-strategi: per-chip 'hvorfor GW X' + risiko-advarsel
- Spiller-modal 'Hvorfor denne spilleren': åpningssetning, gradert rotasjons-
  advarsel, fixture-letthet i klartekst, dødball-oppside
- Arkiv 'Hvor er modellen svak?': ærlig selvkritikk basert på siste 6 GW
  backtest med konkrete bom-eksempler
- Hjem 'Læring fra forrige uke': syntetisert fra prediction-log + backtest
- Kapteinkort FORDI: form gradert, eierskap som differensial-vink, risiko-flagg
- Liga catch-up: eierskap i klartekst + form-evaluering + handling
- Hjem 'Vinnende strategi nå': meta-coaching med 4 sesongfase-scenarioer

**Lag-DNA-pipeline (helt nytt):**

Utnytter `C:\Users\rober\Claude prosjekter\Scraper\output` (147 ark per liga)
til å berike appen med empirisk xG/xGA/CS-data per lag.

- `scripts/build_team_priors.py`: leser tsdl_Premier_League.xlsx +
  tsdl_Bundesliga.xlsx, henter xG/xGA hjemme/borte, CS%, 1./2. omgangs-split,
  BTTS. Output: `data/team_priors.json` (multi-liga v2-format).
- `scripts/update_priors.ps1`: Windows Task Scheduler-script (kjører mandag +
  torsdag 06:30), bygger fresh JSON, committer + pusher hvis endret. Render
  auto-deployer.
- `fpl_optimizer/team_priors.py`: backend-loader med mtime-watching (hot
  reload uten Render-restart), get_league/get_team/league_xg_table/
  team_dna_narrative/fixture_difficulty_from_xg.
- 4 nye API-endepunkter: `/api/team-priors`, `/table?league=PL|BL`,
  `/<team_name>?league=...`, `/fixture-difficulty?home=...&away=...`.

**4 visuelle plasseringer:**

1. Egen **"Lag"-fane** i sidebar (mellom Arkiv og Spiller). Liga-switcher
   PL/Bundesliga, Sterkest+Svakest-hero, full scatter-plot (380px), tabell
   sortert etter net xG.
2. **Mini-DNA i player-modal**: 3 nøkkeltall (Net xG, CS%, 2.-omg-andel) +
   narrativ for spillerens lag.
3. **Top-3/Bunn-3 på Hjem**: kompakt blokk med tre sterkeste og tre svakeste
   PL-lag, lenker til Lag-fanen.
4. **Fixture-tooltip**: hover på top-bar-fixtures gir empirisk xG-prediksjon
   ('Arsenal 2.4 - 0.8 Burnley · FDR 1 vs 5').

**Tall fra dataen (status 2026-05-02):**
- Arsenal: net xG +0.91, CS 47% (sterkest i PL)
- Bayern: net xG +1.81 (sterkest i Bundesliga, mer dominant enn Arsenal)
- Burnley: net xG -1.17 (svakest i PL)

**Endringslogg-tekstene** for 2026-05-01-leveransen omskrevet fra teknisk
jargong til folkelig 'hva fikk du, og hvorfor er det bra'-stil med bullet-
lister.

**Bug-fixes underveis:**
- Liga 'Koble til'-melding viste seg selv om bruker var koblet til
  (`display:flex` overstyrte `[hidden]`-attributtet) — fikset med global
  `[hidden] { display: none !important }`
- Plan-sidens 'Avansert: kjør optimizer fra bunn'-summary roterte teksten
  90° (en CSS-regel i wizardens 'Vis meg hvordan' lekket til alle
  `<details>`-elementer) — scopet til `.wiz-howto`
- Transfer-kort viste £0.0 fordi backend returnerer `cost`, frontend leste
  `price` — fikset med fallback
- PowerShell-script update_priors.ps1 feilet pga em-dashes + 2>&1-redirect
  i PS 5.1 — omskrevet til ASCII-only + & call-operator
- skeleton-state på Liga-leagues-bar (var blank ved cold load)

### Feilsøking 2026-05-02 — pågående

`fpl.kolakowski.no` er live, team-tilkobling fungerer (Nicholas 2006459 lastet
inn med 12 ligaer). Bugs avdekket og fikset:

- **JS-syntaksfeil**: `'This week\\'s call'` brøt template literal → hele
  scriptet feilet å parse → wizardSubmitId og autoDetectFplId ble aldri
  globale. Browser-konsollen var "ReferenceError" overalt. Fix commit `2d3dee9`.
- **1-klikks team-tilkobling**: 3-stegs wizard (Hent → Bekreft → Gå) kollapset
  til ett klikk. Fix commit `ff53926`.
- **404 på /api/leagues og /api/transfer-suggestions**: frontend kalte
  endepunkter som ikke fantes med de navnene. La til backend-aliaser som
  mapper til /api/user (leagues) og suggest_transfers (suggestions).
  Fix commit `56d2ee9`.
- **Rå SVG-tekst i risk-mode-toggle**: `_riskModeLabel` returnerer SVG-string,
  men `labelEl.textContent =` escapet det. Bytter til `innerHTML`. Fix `56d2ee9`.
- **Captain-projection viste 0.8 pts**: leste composite_score (0-1 normalisert)
  i stedet for ekte EP. Bytter til captain_ev / ep_next×2. Fix `56d2ee9`.
- **Tomrom mellom sidebar og innhold på bred skjerm**: `.container { margin: 0 auto }`
  sentrerte i tilgjengelig plass. Bytter til venstre-justert. Fix `aa6cca0`.
- **Roterende "Avansert"-tekst på Plan**: en global CSS-regel i wizardens
  "Vis meg hvordan"-details (`details[open] > summary > span:first-child {
  transform: rotate(90deg) !important }`) lekket til ALLE details-elementer,
  inkludert Plan-sidens "Avansert: kjør optimizer fra bunn"-summary. Scoper
  til .wiz-howto. Fix `b641551`.
- **£0.0 i transfer-kort**: backend-aliaset returnerer player.to_dict() med
  `cost`-felt, mens frontend leste `price`. Legger til fallback. Fix `b641551`.

Gjenstår å feilsøke (neste sesjon):
- Verifisere at deploy `b641551` faktisk fjerner roterende tekst og
  £0.0-bug etter cache-flush
- Se om Hjem-siden renderer riktig nå med Nicholas' data (kaptein, transfer,
  liga-kort)
- Test alle 5 IA-destinasjoner end-to-end

### Hosting & domene — AKTIVERT (2026-05-01)

`https://fpl.kolakowski.no` er live med gyldig Let's Encrypt SSL og GH Actions
cron som treffer riktig URL.

Oppsett:
- **Render-tjeneste:** `fpl-optimizer-e8js.onrender.com` (Free-tier)
- **Custom domain:** `fpl.kolakowski.no` (CNAME hos Loopia/Domeneshop)
- **Keep-alive:** UptimeRobot pinger `/health` hvert 5. min — appen sover aldri,
  så Free-tier oppleves som Starter.
- **GH Actions secret `APP_URL`:** satt til `https://fpl.kolakowski.no`,
  predictions-snapshot cron verifisert grønn (manuell run #2 → success 16s).

Hikker underveis (lærdom for fremtiden):
- Loopia opprettet parkerings-A-records (194.9.94.86/85) ved siden av CNAME-en
  fra "Ingen innstillinger > Parkert"-defaulten. Per DNS-spec ulovlig — løst
  ved å slette og gjenopprette subdomenet med DNS direkte.
- Førsteinntrykk på `fpl.kolakowski.no` viste empty-state-bug: 'Loading fixtures...'
  + 4 tomme stat-kort + duplisert hero/wizard. Fix-pakke (commits 5a5cf3d, 2ee7245,
  e3c6848): `body.not-connected`-klasse satt som default, fjernes kun ved
  bekreftet `loadUser()`-success. Skjuler `#dash-stats`, `#fixture-ticker`,
  `.hjem-page-head` til team-ID er koblet. Hero-card sentrert med 720px max-width
  for å matche wizardens akse.

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
