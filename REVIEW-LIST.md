# REVIEW LIST — things only you (Zaki) should modify or approve

Everything below is ordered by priority. Items marked **[EDIT]** need your real
details; items marked **[DECIDE]** need your call; items marked **[VERIFY]**
need you to confirm the text is accurate before submitting. Deadline: 15 Sep.

---

## A. Must edit before submitting (identity — 10 minutes)

1. **[EDIT] `docs/concept-proposal.tex` line ~214 (§7 Team Capability)**
   Currently: `The lead (Zaki) built and operates the full pipeline end to end...`
   → Replace with your **full name**, add one or two lines of real background
   (degree/affiliation, prior quantum/QI or ML experience) — guidelines §4.1
   explicitly asks for "relevant expertise".
   Then recompile: `cd docs && pdflatex concept-proposal.tex` (twice).

2. **[EDIT] `docs/technical-report-v2.tex` line 32 (author block)**
   Currently: `\IEEEauthorblockN{The qicert Team}` → your real name(s) +
   affiliation line. Recompile twice.

3. **[EDIT] Portal form §4.1 Team Profile**
   Name / role / affiliation / expertise per member, plus lead contact details.
   (Cannot be done in the repo — portal only.)

## B. Decisions only you can make

4. **[DECIDE] Which technical report is "the" report**
   Recommendation: submit **v2 only** (`docs/technical-report-v2.pdf`). v1 is
   kept as history but predates the calibrated-INT8 correction and the search
   results; attaching both invites confusion.

5. **[DECIDE] Submit before or after the confirm stage lands**
   Deadline is 15 Sep. If the confirm result arrives in time, I fold it into
   report §4 (compression table) and proposal §2/§5 in minutes. If you must
   submit first, the docs already label the pending status honestly.

6. **[DECIDE] Run N5 + baseline seeds before submitting?**
   - Seeds 1–2 (~1–2 h GPU) would let the report claim **mean ± std over 3
     runs** — the challenge asks for it explicitly; today we are single-seed
     and the report says so in Limitations (1).
   - N5 (~1.5 h) would activate the degradation-predictor claim in proposal §5
     (currently "planned with metrics fixed").
   - Both fit before the deadline if the GPU queue runs today.

7. **[DECIDE] How to phrase AI-assisted development**
   Proposal §7 currently says AI research agents were used as engineering
   leverage under your direction, with all claims traceable to committed
   artifacts. Read it — if you prefer different wording (or more/less
   prominence), edit that sentence; it is a judgment call about disclosure.

## C. Facts to verify (5-minute read)

8. **[VERIFY] Proposal §3 resource claim: "roughly 15 GPU-hours"**
   My estimate from the ledger (incl. Kaggle T4 rows). Sanity-check against
   `docs/resource-declaration.md`; adjust if your accounting differs.

9. **[VERIFY] Proposal §7 description of who did what**
   Make sure the capability description matches reality as you'd want a judge
   to see it.

10. **[VERIFY] The NO-GO framing is one you can defend out loud**
    Report §4.1 + Table 1 report calibrated INT8 as the accuracy winner at
    2× and the repaired TT route as certificate-sound but below the bar. This
    is the submission's central honesty bet — be comfortable defending it.

11. **[VERIFY] Links render**: repo URL and report path printed on proposal
    page 1; repo public; README displays correctly.

## D. I will do these once you confirm (no action needed from you now)

12. Regenerate `docs/resource-declaration.md` after the final GPU runs
    (it still lists a stale "Tesla T4" machine row from the Kaggle era).
13. Fold the confirm-stage row into report §4 + proposal §5 (if you choose 5a).
14. After seeds 1–2: replace single-seed rows with mean ± std, and update
    Limitations (1) accordingly.
15. Final pre-submit rebuild of both PDFs + page-count/format checks against
    guidelines §5 (≤6 pages proposal ✓, ≤20 MB ✓, ≥10pt ✓, English ✓).

---

*Saved 2026-09-13. Companion: `SUBMISSION-CHECKLIST.md` (portal steps).*
