# Phase-1 Portal Submission Checklist

Use this while filling the portal form. Deadline: **15 Sep 2026**.

## Documents ready

- [x] Concept proposal PDF: `docs/concept-proposal.pdf` (3 pages, guidelines §4.3, all 7 elements)
- [x] Technical report (enclosure, in repo): `docs/technical-report-v2.pdf` (7 pages, claim-to-artifact provenance in Appendix A)
- [x] Public repo: <https://github.com/ZakLr/qicert> (README quickstart, pinned env, 80 tests, experiment log, all artifacts)

## Portal form fields (guidelines §4)

### 4.1 Team profile
- [ ] Team name: `qicert`
- [ ] Lead contact details (name, email — yours)
- [ ] Team members: name / role / affiliation / relevant expertise for each
- [ ] Prior experience with quantum computing, QI methods, or the domain (one
      short paragraph per person; mention the working repo as evidence)

### 4.2 Problem statement selection
- [ ] Select: *Quantum-Enhanced Vision-Language-Action Models in Autonomous
      Driving and Robotics Applications* (Volkswagen Group)
- [ ] One proposal per problem statement — this submission addresses only this one

### 4.3 Concept proposal
- [ ] Upload `concept-proposal.pdf` — verify: ≤ 6 pages ✓ (3), PDF ✓, ≤ 20 MB ✓, English ✓, ≥ 10pt ✓

### 4.4 Optional supplementary material
- [ ] Optional: up to 3 appendix pages — already inside the repo; the
      technical report serves as the supplementary artifact via the repo link
- [ ] Repo link in the form (it is also printed inside the proposal PDF)

## Before you click submit

- [ ] Open the PDF once on your machine and read pages 1–3 start to finish
- [ ] Fill in your real name/contact in `docs/concept-proposal.tex` §7
      (currently "Zaki" placeholder in Team Capability), recompile
      (`pdflatex concept-proposal.tex` twice), re-upload
- [ ] Team profile §4.1 fields completed above
- [ ] Repo is public and the README renders (last check: `bf1da9e`)
- [ ] The confirm-stage result is folded into the technical report §4
      (compression table) — if not yet, note the pending row is labeled as such
- [ ] Repo contains no confidential third-party information (guidelines §3/§4.4)

## Where each guidelines element lives in the proposal

| Guidelines §4.3 element | Proposal section |
|---|---|
| Problem Framing | §1 |
| Technical Approach (paradigm stated) | §2 |
| Feasibility and Resource Requirements | §3 |
| Expected Impact (quantitative targets) | §4 |
| Validation Plan (metrics = success) | §5 |
| Hybrid / Cross-Domain Integration | §6 |
| Team Capability | §7 |
