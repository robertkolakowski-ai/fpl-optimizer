# FPL Optimizer — Backlog

**North star:** Fantasy Premier League. Gode råd. Hjelpe brukeren oppnå best mulig resultater.
Hver feature måles mot: *gjør den brukerens FPL-rang bedre, eller gjør den ikke det?*

---

## Park / Deferred

### Hosting & domene
Flytt fra Render free → en plattform som er alltid på, tilgjengelig på `fpl.kolakowski.no`.
Beslutning om plattform tas senere (Render Starter / Fly.io / Railway / annet).
Ikke blokkerende for redesign — alle kodeendringer fungerer uavhengig.

### Spiller-modal (full)
Konseptbrief Del 4.7. Klikk på en spiller hvor som helst → modal/full-screen
med projeksjon-graf, set pieces, rotasjonsrisiko, sammenligning, Advanced tab
med Opta-stats. I dag finnes bare en search-side. Ikke startet.

### Real predictions log
Persistere modellanbefalinger per GW for ekte hit-rate i Arkiv. Krever
DB eller fil-skriving for å bygge troverdig "Hadde modellen rett?"-loop.
Inntil dette finnes er Arkiv-statistikk merket som heuristikk.

### Web Push (server-side)
Send faktisk varsel mandag morgen via cron + Push-API. Bare opt-in-flow
finnes nå (browser-permission, lokal flag). Trenger backend-job.

### Sammenligning hvor som helst
Generell "sammenlign to spillere"-mekanisme tilgjengelig fra Hjem og Plan,
ikke bare Kaptein. Konseptbrief Del 03 bullet 19.

### Touch drag-and-drop på Plan-pitch
Mobil-versjon. HTML5 drag-events fungerer ikke godt på touch — trenger
egne touchstart/touchmove/touchend-handlers. (G3 dekker minimum mobil-
versjon, men full drag-drop er parkert.)

### Liga line-chart caching
Trend-chart fetcher 5 brukere parallelt fra `/api/user/<id>` per visning.
Treg + ikke cachet. Trenger localStorage TTL (10 min) + bedre batch-API.

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
