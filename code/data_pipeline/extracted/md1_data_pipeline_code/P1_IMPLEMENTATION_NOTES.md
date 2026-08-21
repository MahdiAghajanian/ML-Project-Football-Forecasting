# P1 implementation notes — final TA-approved design

## Decision

The former multi-season La Liga default is retired as the primary P1 path. The final implementation uses the TA-approved hybrid EPL design:

- Football-Data EPL 2000/01--2014/15: basic final-result history only;
- StatsBomb EPL 2015/16: target/modeling season and current-season causal continuation;
- Football-Data EPL 2015/16 odds: independent market benchmark in the existing baseline pipeline.

## Why this is defensible

Berrar requires complete same-league longitudinal history. Historical Football-Data rows are used as context, not as final supervised examples. The final P1 matrices contain one row per StatsBomb EPL 2015/16 target match, inherit the same chronological split semantics as MD1, and select recency using training labels only.

## Paper fidelity categories

- **Exact:** league-specific Super League continuity, same-league participation gaps, mean aggregation, minimum six prior matches, 6/18 feature contracts, scored/conceded/league-success core.
- **Paper-aligned with documented ambiguity:** odd `n` venue count uses `floor(n/2)`; normalized-rank operational tie behavior is made explicit.
- **Project-required adaptation:** TA-approved historical source, StatsBomb-only final examples, chronological outer split, train-only recency selection, Task R margin target.
- **Project-specific extension:** Berrar pre-match vector may be inherited into Task L snapshots.

## Important anti-contamination rule

The historical parser explicitly selects only:

```text
Date, HomeTeam, AwayTeam, FTHG, FTAG
```

Any odds, shots, cards, corners, referees, halftime fields, or other Football-Data columns are discarded before the canonical P1 history table is created.

## Target-season history

Football-Data 2015/16 final results are not used for P1. During the target season, earlier completed matches enter later P1 histories from StatsBomb itself. This keeps target-season labels and causal continuation on the primary project provider.
