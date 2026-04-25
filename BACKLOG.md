# FPL Optimizer — Backlog

**North star:** Fantasy Premier League. Gode råd. Hjelpe brukeren oppnå best mulig resultater.
Hver feature måles mot: *gjør den brukerens FPL-rang bedre, eller gjør den ikke det?*

---

## Etter Fase 4f — kritisk

### Hjem-cleanup (kaos-pass)
- **Symptom:** Robert ser at Hjem-siden er "totalt kaos" på live deploy etter Fase 4b/4c/4d.
- **Sannsynlig årsak:** Phase 2 light-theme override-laget kolliderer med ny Amber/Ink/Paper-identitet fra 4a; gamle CSS-regler med hardkodede mørke farger maskerer ny token-styling; Fraunces variable font (`opsz` axis) lastes ikke som forventet; legacy nav-panel-stil sniker seg inn under nye komponenter; kontrast-issues mellom Hjem-kort og Paper-bakgrunn.
- **Løsning:** Egen "Hjem-cleanup"-fase — rip ut Phase 2 override-laget, refaktorer alle Hjem-komponenter til å bruke kun :root-tokens, audit alle hardkodede farger på Hjem-siden, sjekk at Fraunces faktisk laster. Sannsynligvis 1 commit, 100-200 linjer.
- **Skal gjøres rett etter Fase 4f.**

---

## Deferred / parked

### Hosting & domene
- Flytt fra Render free → en plattform som er alltid på, tilgjengelig på `fpl.kolakowski.no`.
- Beslutning om plattform tas senere (Render Starter / Fly.io / Railway / annet).
- Ikke blokkerende for redesign — alle kodeendringer fungerer uavhengig av hvor det deployes.

---

## In progress: 25-punkts redesign

Aktiv plan i 4 faser. Hver fase = én commit å reviewe.

- **Fase 1 — Fundament & arkitektur** (#3, #4, #6, #25): Team ID-gate, Plan/Live/Review, i18n-skjelett, premium-flagg. *(nå)*
- **Fase 2 — Designsystem** (#7, #8, #9, #13): Lys høy-kontrast tema, ny primærfarge, typeskala, mobile-first.
- **Fase 3 — Hjem & forklarbarhet** (#1, #2, #14, #15, #18, #20, #21, #22): USP-hero, 3-stegs onboarding, hjem-brief, "fordi"-felt, side-panel m/ forklaring, glossar.
- **Fase 4 — Interaksjon & innhold** (#10, #11, #12, #16, #17, #19, #23, #24): Spillerkort, mikro-viz, ekte pitch m/ drag-drop, sliders, sammenligningsmodus, ukerapport.
