# NYC reproduction: summary

Evaluation every 2,000 steps, greedy (epsilon = 0), 250k steps, frozen graph `data/nyc`. AUC = mean evaluation return over the run, with a 95% bootstrap CI over seeds.

| method | n | AUC (mean eval return) | sigma of AUC | final return (last 20k steps) | seeds ever solving eval | first solved eval (k steps, median) | final coverage (nodes) | wall clock (min, mean) |
|---|---|---|---|---|---|---|---|---|
| NEO | 20 | 52.2 [41.4, 62.9] | 24.3 | 88.5 | 19/20 | 88 | 8444 ± 1436 | 2.3 |
| NEO (first ev.) | 20 | 45.5 [34.4, 56.7] | 25.3 | 79.5 | 19/20 | 100 | 8354 ± 1448 | 1.6 |
| SP-novelty | 20 | 1.8 [0.0, 4.4] | 5.1 | 0.0 | 4/20 | 10 | 5024 ± 1528 | 1.8 |
| Q-learning-novelty | 20 | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0/20 | - | 2284 ± 504 | 0.9 |

## Paired comparisons (same seed = same start/goal), Wilcoxon signed-rank on AUC

| comparison | n | mean AUC difference | p |
|---|---|---|---|
| neo vs sp | 20 | +50.4 | 0.00013 |
| neo vs qlnov | 20 | +52.2 | 0.00013 |
| neo vs neo1 | 20 | +6.7 | 0.33 |

## NEO eigenvalues at option updates (all seeds)

| variant | eigenvalues | complex (abs(Im) > 1e-8) | max abs(Im) | Re < delta = 0.5 | min Re | max Re |
|---|---|---|---|---|---|---|
| neo | 320 | 8 | 1.0e-03 | 320 | 0.4515 | 0.4808 |
| neo1 | 80 | 0 | 0.0e+00 | 80 | 0.4515 | 0.4750 |

## AUC by goal node (seeds sharing a goal are not independent instances)

| goal | seeds | NEO | NEO (first ev.) | SP-novelty | Q-learning-novelty |
|---|---|---|---|---|---|
| 7367 | 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 18 | 57.7 | 44.6 | 1.4 | 0.0 |
| 2181 | 2, 16, 17, 19 | 45.2 | 60.8 | 4.8 | 0.0 |
| 5024 | 0, 9, 14 | 37.9 | 29.1 | 0.0 | 0.0 |
