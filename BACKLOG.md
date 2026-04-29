# FPL Optimizer — Backlog

**North star:** Fantasy Premier League. Gode råd. Hjelpe brukeren oppnå best mulig resultater.
Hver feature måles mot: *gjør den brukerens FPL-rang bedre, eller gjør den ikke det?*

---

## Park / Deferred

### Hosting & domene — KLAR FOR AKTIVERING
Status: kode + cron-workflow er på plass. Gjenstår beslutning + DNS.
Anbefalt sti: Render Starter ($7/mnd, ingen cold-start) + custom domain
`fpl.kolakowski.no`.

Steg for aktivering:
1. Render dashboard → tjenesten → Settings → Instance Type → **Starter**
2. Settings → Custom Domains → legg til `fpl.kolakowski.no` → Render gir
   en CNAME-verdi
3. DNS hos kolakowski.no-registrar: CNAME `fpl` → den onrender.com-verdien
4. GitHub repo → Settings → Secrets → Actions → ny secret `APP_URL` =
   `https://fpl.kolakowski.no` (kreves for predictions-snapshot cron)
5. Verifisering: `curl -X POST https://fpl.kolakowski.no/api/predictions/snapshot`
   skal returnere `{"ok": true, ...}`

Alternativ: Fly.io Hobby (krever Dockerfile + Fly CLI) eller bli på Render
Free (cold start på første besøk).

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
