# Deviations from the original setup

Every change relative to the authors' code (tag `upstream-original`, commit `bfa5aa1`) or to the paper's text, why we made it, and what it may change.
Scope so far: C2 / C3 (NYC). Numbers refer to [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md) §3.

| # | What | Original | Ours | Why | Possible effect |
|---|---|---|---|---|---|
| D1 | Indentation fix in `all_functions.py:594` | TabError, module does not import | spaces | code did not run (§3.6) | none (whitespace) |
| D2 | Notebook → scripts | `NYC_Street_Evals.ipynb` | `Many_Streets_Eval.py` (Céline) + `run_nyc.py`; `run_training_on_env` logic unchanged, only a `seeds=` argument added | one process per (method, seed), parallel runs, saved results | none intended |
| D3 | Frozen graph | downloaded live from OSM on every run | downloaded once on 2026-11-10 with the same settings, stored in `data/nyc/` (`nyc_graph.py`) | OSM changes over time (§3.8) | our graph: 10,000 nodes / **22,322** edges vs. paper 10,000 / 22,354 (−0.14 % edges); not the authors' exact graph |
| D4 | Software versions | not reported | Python 3.12.7, numpy 2.5.4, scipy 1.18.1, networkx 3.7, osmnx 2.1.1 (`requirements.txt`) | no pins upstream | ARPACK / BFS order may differ from the authors' |
| D5 | NumPy seeding | only `random.seed(run)` (start/goal) | additionally `np.random.seed(seed)` at process start | ε-greedy, option choice and env use NumPy; unseeded = not reproducible (§3.9) | none in expectation; makes runs repeatable |
| D6 | Number of runs | code default `instances=1` | 20 seeds (0–19) | paper reports 20 instances (§3.7) | — |
| D7 | Start/goal selection | paper: random goal, start = farthest from goal | **kept the code's rule**: random start, goal = farthest reachable node (hops) | the code produced Fig. 3k | the 20 seeds use only **3 distinct goals** (node 7367 for 13 seeds, 2181 for 4, 5024 for 3): seeds are not 20 independent tasks |
| D8 | Error band | ±2·std over seeds (`TWO_SIGMA` in `chart_utils.py`) | 95 % bootstrap CI of the mean; std reported in the table | return is 0/100 per evaluation, so ±2σ leaves [0, 100] | presentation only |
| D9 | Saved results | (saving commented out) whole result dict incl. final agent | metrics only (eval returns, coverage, eigenvalues, visitation, wall clock) | each option deep-copies the agent's graph → very large files | none |

## Open questions (not yet decided)

- **SP-novelty options per update:** the NYC notebook uses 4 (four least-visited targets), `plots.ipynb` labels SP-novelty with 1. We use 4.
- **Novelty bonus:** main text and code use 0.01/√n(s,a); appendix Table 3 says β/√N(s). We use the code's version.
- **Option Q-initialisation:** paper −1e-5, code −0.001 for not-yet-updated options. We keep the code's.
