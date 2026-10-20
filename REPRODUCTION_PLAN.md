# Reproduction Plan: *Novel Exploration via Orthogonality* (NEO)

**Paper:** A. Theophilou & Ö. Şimşek, *Novel Exploration via Orthogonality*, NeurIPS 2025 ([OpenReview](https://openreview.net/forum?id=yJS1eZSNUv)).
The PDF was removed from the repo at HEAD. You can recover it with `git show dc4de29:Novel_Exploration_via_Orthogonality.pdf > NEO_paper.pdf`.
**Group:** Céline Wallart, Nathan Brack, Yilin Lin · **Supervisor:** Lucia Pezzetti
**Course:** UZH Reinforcement Learning, Reproducibility Project, Fall 2026 · **Poster session:** 15 Dec 2026
**Status:** draft, 20 Oct 2026

---

## 1. Scope

In her email of 13 Oct, Lucia asked us to prioritise three items. These are the core of the project:

| # | Claim / question | Type | Priority |
|---|---|---|---|
| **C1** | **Theorem 3.3** (novelty path via visit counts): reconstruct the proof, check every assumption and every strict inequality, and stress-test the statement with small counter-example candidates (zero eigenvector entries, plateaus). | Theory + small simulations | Core |
| **C2** | **NYC result (Fig. 3k):** NEO outperforms Q-learning-novelty and SP-novelty on the directed NYC street graph. We use repeated runs and a written evaluation protocol. | Empirical reproduction | Core |
| **C3** | **NEO (first eigenvector only) on NYC.** Fig. 3k lacks this variant. Does using several eigenvectors help in this environment, as the paper claims for the other domains? | Ablation / extension | Core |
| S1 | **Eigen-options failure on NYC:** find exactly what fails, and test the paper's explanation (polar decomposition plus weakly connected graphs). | Analysis | Stretch, **timeboxed** |
| S2 | **Second city:** NEO on a city graph chosen for a specific structural difference from NYC. | Generalisation | Stretch, **timeboxed** |
| — | Runtime optimisation | Engineering | Optional; only if it unblocks C2 |

We report reproduction (C1, C2) and extensions (C3, S1, S2) in separate sections, as the course guidelines ask.

---

## 2. What the paper claims (to be tested)

**Theorem 3.3 (main text, p. 4).** Let G be strongly connected and N(i) ≥ 0 a visitation value. Define L_ζ = (I + Γ) − P with Γ_ii = N(i). For an eigenpair (λ, f) and a node with N(i) > Re(λ), there is a directed path v₀ → … → v_k with |f_{v₀}| < |f_{v₁}| < … < |f_{v_k}| and **N(v_k) < Re(λ)**.

The appendix gives an extended version (Thm 8.1) that splits into three cases:
- Cases 1–2 (λ real): the conclusion is N(v_k) < λ.
- Case 3 (λ complex): the conclusion is only **N(v_k) ≤ Re(λ)**.

**NYC experiment (§6, Fig. 3k).** The domain is a directed street graph (10,000 nodes, 22,354 edges) built from OpenStreetMap around NYC City Hall. The paper reports that NEO beats the alternatives, and that eigen-options "could not be computed because the solver would crash".

---

## 3. Issues found during a first read of the paper and code

Each item below is either something to verify (V) or a deviation we must document (D). Nothing here is a conclusion yet.

### Theory (C1)
1. **(V) Strictness of the endpoint bound.** The main text claims N(v_k) < Re(λ) for complex λ. The appendix proves only ≤ in Case 3. In Cases 1–2, strictness is argued "by strong connectivity". This argument is not obviously complete: equality N(v_k) = λ gives z = 1, which seems possible when all out-neighbours of v_k have the same |f| (a plateau).
2. **(V) Zero entries.** The first strict step |f_i| ≤ (1/|z_i|)·Σ_j P_ij|f_j| < Σ_j P_ij|f_j| needs Σ_j P_ij|f_j| > 0. If f_i = 0 and all out-neighbours are 0, the chain never starts. The appendix assumes f_i ≠ 0; the main-text statement does not.
3. **(V) Plateaus and the algorithm.** The theorem guarantees a *strictly* increasing path. The NEO option, however, terminates at any node where |f| ≥ |f| of all neighbours, ties included (`take_option_action`, [all_functions.py:389](all_functions.py#L389)). An option can therefore stop on a plateau before reaching a node with low N. We need to check how far the guarantee actually transfers to the option policy.
4. **(V) Assumptions vs. practice.** The paper itself says the agent's graphs "remained weakly connected throughout most of learning", so the strong-connectivity assumption does not hold where NEO is used. In the code, nodes with no observed out-edge (the exploration frontier) get an **all-zero row in P** ([utils.py:74](utils.py#L74)). P is then not row-stochastic, and from row i of L_ζ f = λf it follows that f_i = 0 unless λ = 1 + ζ_i. If this holds in practice, NEO options could never point *at* frontier nodes. To check: which steps of the proof survive with sub-stochastic rows or without strong connectivity?
5. **(V) The theorem uses raw N; the algorithm uses ζ = F(N) = δ·(N/max N)^{1/k}** with δ = 0.5 and k = 64, and keeps eigenvectors with λ < δ. We need to confirm that the algorithm's condition maps cleanly onto the theorem's hypothesis. The prose after Remark 3.1 also states the gradient direction backwards ("from lower to higher visitation"); this is minor, but we should note it.

### Code (C2/C3)
6. **(D) The code does not run as published.** `all_functions.py` fails to parse: `TabError` at [line 594](all_functions.py#L594), introduced in the last code commit. There is no `requirements.txt` and no version pins. The import `sklearn.dummy.check_random_state` may also fail on recent scikit-learn.
7. **(D) Number of runs.** `run_training_on_env` in `NYC_Street_Evals.ipynb` defaults to `instances=1`, and `main()` does not override it. The paper reports 20 instances. Saving the results is also commented out.
8. **(D) The NYC graph is not frozen.** It is downloaded live from OpenStreetMap with `osmnx` (7 km radius, largest *weakly* connected component, then a 10k-node BFS sample). Today's OSM data will give a different graph. We will snapshot it once, commit the file, and compare its node/edge counts with the paper's 10,000 / 22,354.
9. **(D) Start/goal selection differs from the paper.** The paper picks a random goal and starts at the farthest state from it. The NYC code picks a random start (with `random.seed(run)`) and sets the goal to the farthest *reachable* node. NumPy's RNG is never seeded, so ε-greedy behaviour cannot be reproduced exactly.
10. **(D) Novelty bonus.** The main text and code use β/√n(s,a). Appendix Table 3 says β/√N(s).
11. **(D) Option Q-initialisation.** The paper says −1e-5; the code sets −0.001 for options whose Q-value has not been updated yet.
12. **(D) Error bands.** `chart_utils.py` with `TWO_SIGMA` plots **±2·std** across seeds, which shows the spread of runs and is not a confidence interval. We will report both the spread and a 95% CI.
13. **(V) Few option updates.** With H = 5 × 10,000 = 50,000 and a 250k-step budget, options are recomputed only about 4 times per run. This matters when we interpret C3.
14. **(V) Possible option-index bug.** Options live in a `deque(maxlen=64)`. When it is full, appending shifts every index, but the Q-table option columns are not shifted. Old Q-values could then attach to different options. Needs a check.
15. **(V) Dense matrices.** `add_option` builds a dense 10k × 10k adjacency matrix (about 800 MB per copy). For eigen-options it then runs a dense `scipy.linalg.polar` (an SVD, O(n³)). A memory or time failure is a competing explanation for the "crash" (see S1).

---

## 4. Work packages

### WP0: Setup (all; week of 20 Oct)
- [x] Set up the fork: `origin` = [LYL010-ADELA/NEO-RL26HS](https://github.com/LYL010-ADELA/NEO-RL26HS), `upstream` = AndreasTheo/NEO. The untouched upstream commit `bfa5aa1` is tagged `upstream-original` (local tag; push it with `git push origin upstream-original`).
- [ ] Minimal fix to make the code import (TabError). Every code change goes into a **deviations log** (`DEVIATIONS.md`) that records what changed, why, and its possible effect.
- [ ] Pin the environment (`requirements.txt` or `environment.yml` with Python, numpy, scipy, networkx, osmnx, scikit-learn versions).
- [ ] Convert `NYC_Street_Evals.ipynb` into a script with CLI arguments (`--method --seed --num-options --graph`). The script saves per-run results (eval returns, node/edge counts, eigenvalues, option info) to `results/`. This lets us run seeds in parallel and restart failed ones.
- [ ] Snapshot the NYC graph (`data/nyc_graph.graphml`) and record the download date and osmnx version. Report: #nodes, #edges, number of strongly connected components and size of the largest, out-degree-0 nodes, and the fraction of (start, goal) pairs where the goal is reachable.
- [ ] **Pilot:** 1 seed × each method, with wall-clock time and peak memory measured. Then extrapolate: total ≈ 4 methods × 20 seeds × T_run. Use this to decide whether we need RenkuLab compute. If we do, one email to kone@ifi.uzh.ch.

### WP1: Theorem 3.3 (C1)
1. **Restate precisely.** Write the hypotheses in full, including the implicit ones: row-stochastic P, f_i ≠ 0, every node has an out-edge. Write separate statements for the three cases (λ real/complex, f real/complex).
2. **Reconstruct the proof ourselves** (not a paraphrase of the appendix). Make a table with one row per step, showing which assumption it uses and which inequality is strict and why.
3. **Probe the weak points** from §3 (items 1–5):
   - Does N(v_k) < Re(λ) hold strictly, or only ≤? Try to build an example with equality.
   - Zero entries: look for eigenvectors that vanish on a start node and all its out-neighbours. Candidates are symmetric graphs with symmetric N, where antisymmetric eigenvectors are 0 on the symmetry axis.
   - Plateaus: graphs with automorphisms that give equal |f| on neighbouring nodes.
   - Relaxed assumptions: sub-stochastic rows (frontier nodes), weakly connected graphs, dangling nodes. Does the statement still hold, possibly in a modified form?
   - Complex eigenvalues: directed cycles and random digraphs with non-uniform N.
4. **Numerical checker** (`theory/check_thm33.py`). For a given graph and N: compute all eigenpairs. For each eigenpair and each node meeting the hypothesis, search (DFS over strictly increasing |f|) for a path that ends at a node with N < Re(λ), and log any failure. Use an explicit tolerance and report sensitivity to it.
   Test graphs: directed cycle, path, small grids, directed four-rooms, random (di)graphs, and **empirical NYC graphs/visit counts saved during WP2 runs**. The last ones also show whether the theorem's hypotheses hold where NEO is actually used.
5. **Output:** a write-up (LaTeX/Markdown) that marks each part as *verified*, *verified with an extra assumption*, *counter-example found*, or *open*.

Suggested split: one person leads, and a second person independently checks the proof and builds the counter-examples.

### WP2: NYC reproduction (C2)
**Methods:** NEO (Z = 4), SP-novelty (`oracle` in code), Q-learning-novelty. NEO-first-ev belongs to WP3 but runs in the same sweep.

**Protocol** (fixed before the main runs, copied into the report):

| Item | Value (paper / code) | Note |
|---|---|---|
| Agent | Q-learning, α = 0.4, γ = 0.99, ε = 0.1, reverse-episode replay | |
| Options | initP = 0.1; 4 new per update; cap 64; SMDP Q-updates | NEO-first-ev: 1 per update |
| Option discovery period H | 5 × #nodes = 50,000 steps | about 4 updates per run (§3.13) |
| NEO scaling | δ = 0.5, k = 64 | |
| Novelty bonus | 0.01/√n(s,a) | main text vs. appendix (§3.10) |
| Episode horizon | #nodes = 10,000 | |
| Budget | 250,000 steps | |
| Evaluation | every 2,000 steps, one greedy episode (ε = 0, no option exploration) | |
| Seeds | **20** (seed = run index), same start/goal across methods | code default is 1 (§3.7) |
| Start/goal | as in the NYC code (random start, farthest reachable goal) | differs from paper text (§3.9); document it |
| Graph | frozen snapshot (WP0) | §3.8 |

**Metrics:**
- Evaluation return curve: mean with ±2·std (the paper's band) and a 95% bootstrap CI.
- Fraction of seeds that reach the goal at least once, and steps to first goal.
- Area under the evaluation curve.
- Unique nodes and edges visited over time (coverage, as in Fig. 4).

**Statistics:** paired comparisons across seeds (same start/goal), using a Wilcoxon signed-rank test on AUC and final return. Effect sizes are reported alongside.

**Success criterion (set in advance):** the reproduction counts as successful if NEO ranks above both baselines on AUC with a CI that excludes zero difference, and the curves look qualitatively like Fig. 3k. Partial or failed reproduction is a valid outcome; we will report it as such.

**Also log, per option update:** the Z eigenvalues (real or complex, and whether they are < δ), the visitation counts at option termination states, and option lengths. This connects WP1 to WP2.

### WP3: NEO with only the first eigenvector (C3)
- Same protocol as WP2 with `num_options = 1` (as `NEO(first-ev)` in `plots.ipynb`).
- Compare NEO (Z = 4) and NEO (Z = 1) on return, coverage, and termination-state novelty.
- Confounder: with Z = 1 the agent gets four times fewer options per update, so the option library fills more slowly. If time allows, add a control that holds the number of options fixed, e.g. Z = 1 refreshed four times as often, or Z ∈ {1, 2, 4, 8}.

### S1: Eigen-options failure on NYC (stretch, timebox ≈ 1 person-week)
1. Reproduce the crash and record the exact error, the step at which it happens, and memory use.
2. Separate the candidate causes:
   - (a) memory/time from the dense O(n³) polar decomposition;
   - (b) numerical instability with weakly connected graphs (the paper's hypothesis);
   - (c) ARPACK `eigs` not converging.
   Tests: run on growing subgraphs; restrict to the largest strongly connected component; compare dense and sparse paths.
3. **Stop rule:** if the cause is not identified by the end of the timebox, we report what we ruled out.

### S2: Second city (stretch, timebox ≈ 1 person-week, decide by ~17 Nov)
- Pick the city for one structural contrast with NYC. Examples: a much higher or lower share of one-way streets; an organic, non-grid street layout (e.g., an old European city centre); a different number or size of strongly connected components.
- Reuse the exact WP0 pipeline (snapshot, same size of about 10k nodes, same protocol). Run only NEO, NEO-first-ev, and the cheaper baseline if compute is tight.
- Go/no-go at the mid-November meeting, based on how far WP2 and WP3 have progressed.

---

## 5. Timeline

| Week | Dates | Theory (WP1) | Empirical (WP0, WP2, WP3) |
|---|---|---|---|
| 1 | 20–26 Oct | Restate theorem; start proof reconstruction | Fork, fix import, pin env, freeze NYC graph, script-ify notebook |
| 2 | 27 Oct – 2 Nov | Step-by-step proof table; build checker | Pilot runs + timing → compute decision; finalise protocol |
| 3 | 3–9 Nov | Counter-example search (zeros, plateaus, equality case) | Launch 20-seed sweep: NEO, NEO-1, SP-novelty, QL-novelty |
| — | **Tue 10 Nov** (proposed) | **Meeting with Lucia: first results** | |
| 4 | 10–16 Nov | Relaxed assumptions (frontier/weak connectivity); run checker on NYC snapshots | Analysis + plots; follow-up runs; S1 timebox starts |
| 5 | 17–23 Nov | Write up theory results | S1 ends; S2 go/no-go and runs if go |
| 6 | 24–30 Nov | Buffer / integrate | Final runs, statistics, figures |
| 7 | 1–7 Dec | Report (MLRC format) | Reproducibility package (README, env, commands, seeds) |
| 8 | 8–14 Dec | Poster + rehearsal (everyone must be able to explain every part) | |
| — | **15 Dec** | **Poster session** | |

**For the 10 Nov meeting, bring:**
- the proof-step table and any counter-example candidates;
- the NYC graph statistics;
- the pilot and early-seed learning curves for all 4 methods;
- the list of deviations so far.

---

## 6. Division of work (proposal, to be agreed)

| Role | Main tasks | Owner |
|---|---|---|
| Theory lead | WP1 proof reconstruction + write-up | TBD |
| Theory checker / simulations | WP1 checker, counter-examples; links to WP2 logs | TBD |
| Experiments lead | WP0 pipeline, WP2/WP3 runs, S1 | TBD |
| Shared | Deviations log, statistics, report, poster | all |

Everyone reviews everyone else's part before the 10 Nov meeting and again before the report is submitted.

---

## 7. Deliverables and repository layout

```
NEO/
├── REPRODUCTION_PLAN.md   ← this file
├── DEVIATIONS.md          ← every change vs. upstream + rationale
├── requirements.txt
├── data/                  ← frozen graph snapshots (+ download metadata)
├── scripts/run_nyc.py     ← CLI: method, seed, num_options, graph
├── theory/                ← proof write-up + check_thm33.py
├── results/               ← per-run outputs (not plots)
└── analysis/              ← notebooks that make every figure/table from results/
```

The final report (MLRC style) will cover: original claims, our methodology, deviations, results with uncertainty, limitations, and conclusion. The poster is due 15 Dec.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Runs too slow (20 seeds × 4 methods × 250k steps, dense 10k × 10k matrices, deepcopy at every eval) | Pilot first; run seeds in parallel; RenkuLab if needed; as a last resort, reduce seeds and say so explicitly |
| Our NYC graph differs from the authors' | Snapshot and report the differences; optionally ask the authors for their graph |
| Hidden code/paper mismatches change the results | Default to the code's behaviour (that is what produced Fig. 3k) and document each mismatch; check the important ones with ablations |
| Theory counter-example search goes nowhere | Fine as an outcome: report which cases were checked and how |
| Stretch goals eat into the core work | Hard timeboxes; S2 needs an explicit go at the mid-Nov meeting |
