# Instructions for coding tools

Read PROJECT_PLAN.md fully, then PROGRESS.md (create it if missing). One milestone per session, in
order (Section 7). Tests first. Keep code readable.

Hard rules:
- No leakage; never tune on the holdout; the holdout runs once from committed, frozen code.
- No invented figures; every number in docs/report comes from saved results.
- Assumptions and scope live in config/ with a reason; never hard-code them.
- Do not commit data/raw/ or any secret. Pin dependencies in requirements.txt.
- 8 GB RAM laptop: use the configured slice, DuckDB and parquet.
- Mark uncertain things TO VERIFY in NOTES.md. End of session: update PROGRESS.md and add a short
  plain-language note in docs/. Commit nothing unless asked.
