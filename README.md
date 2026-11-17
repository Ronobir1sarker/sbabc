# SBABC: a set-based artificial bee colony algorithm

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24326043.svg)](https://doi.org/10.5281/zenodo.24326043)

Code, tuned configurations and results for the paper

> R. C. Sarker and J. C. Bansal, *A Generic Set Based Artificial Bee Colony
> Algorithm for Combinatorial Optimization*, submitted to Swarm and
> Evolutionary Computation.

SBABC keeps the employed, onlooker and scout phases of the artificial bee
colony algorithm (ABC) and builds every move from set operations: a bee moves
toward a neighbouring food source, drops elements that the two share and adds
elements that neither holds, chosen by tournament. The search reads only the
objective value and the constraint violation of a solution, so the same code
runs on any subset-selection problem.

The repository contains every algorithm of the paper, the four problem
domains with their instances, the tuning protocol, the experiment driver and
the scripts that reproduce each table and figure.

## Installation

Python 3.13 and the exact versions in `requirements.txt` are required for
bit-identical results: every run draws its random numbers from a seed derived
from the experiment seed and the run index, and the results database records
the library versions it was produced with.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q          # a few minutes
```

All commands run from the repository root.

## Reproducing the paper

The per-run results of every experiment are stored in one SQLite database
(about 1.8 GB), archived on Zenodo:

> **Database:** [doi:10.5281/zenodo.24326069](https://doi.org/10.5281/zenodo.24326069)
> (`results.sqlite.xz`, 80 MB; `SHA256SUMS.txt` gives the checksums)

Download it, decompress it to `results/results.sqlite`
(`xz -d results.sqlite.xz`) and run the script for the table you want. Every script opens the database read-only.

| Paper | Script |
|---|---|
| Table 3, tuned configurations | `paper-experiments/tuned_config_check.py` (needs only `results/tuning/`) |
| Tables 4–8 and 10, knapsack | `paper-experiments/mkp_tables.py` (also `large9_check.py`) |
| Table 9, operator ablation | `paper-experiments/ablation_rev2_check.py`, `tie_rule_check.py` |
| Tables 11 and 12, maximum diversity and feature selection | `paper-experiments/domain_tables.py --domain mdp` / `--domain fs` |
| Table 13, set covering | `paper-experiments/scp_boundary_check.py` (needs only `results/tuning/`) |
| Table 14, consistency across benchmarks | `paper-experiments/consistency_check.py` |
| Table 15, parameter sensitivity | `paper-experiments/sensitivity_table.py` (needs only `results/tuning/`) |
| Table 16, constraint handling | `paper-experiments/penalty_check.py` |
| Section 8.5, one-at-a-time check | `paper-experiments/sensitivity_oat_check.py` |
| Figures 2 and 3 | `paper-experiments/fig_convergence.py`, `fig_walltime.py` |
| Statistics behind every significance mark | `paper-experiments/significance_export.py` |

The files already in `results/` need no database:

- `results/tuning/`: for each domain and algorithm, the 32 sampled
  configurations with their tuning errors and mean rank, and the selected
  configuration (`<domain>_best_configs.json`).
- `results/significance/`: every pairwise comparison behind a significance
  mark (1919 comparisons), with the paired Wilcoxon p-value, its Holm
  adjustment, Cliff's delta and a bootstrap 95 % confidence interval of the
  mean difference. Pairs of runs with equal values are discarded, as in the
  original test.
- `results/*.csv`: per-instance summaries of the main experiments, and the
  Weing reproduction runs of ABABC and BGWO.

## Rerunning the experiments

Tuning comes first, then the experiments. Both are resumable: every finished
run is written to the database at once, so rerunning a command continues
where it stopped.

```bash
./run_fast.sh run_tuning.py --domain mkp --workers 15        # domains: mkp scp fs mdp
./run_fast.sh run_experiment.py --experiment mkp_competitors_rev2 --workers 15
./run_fast.sh run_experiment.py --experiment mkp_competitors_rev2 --summary
```

`run_fast.sh` sets one BLAS thread per worker. The experiments of the paper
are registered in `run_experiment.py` (`CAMPAIGNS`); the main ones are
`mkp_competitors_rev2` (knapsack, 64 instances), `mkp_large90_rev2`
(90 Chu–Beasley instances), `mkgk_rev2` (Glover–Kochenberger instances),
`mkp_cb_rev1` (repair-equipped variants), `mkp_ablation_rev1` and
`mkp_ablation_rev2` (ablation), `mdp_main_rev1`, `fs_main_rev1` and
`mkp_penalty_rev2`. The set covering results come from the tuning runs
(`run_tuning.py --domain scp`). With `--runs 0-14` an experiment can be split
across machines and merged later with `--merge`.

## Layout

```
core/              Optimizer base class and Config: run loop, topologies, Deb's rule,
                   evaluation counting, per-run seeding
algorithms/        SBABC and the eight comparators (SBPSO, AMABC, BitABC, GBABC,
                   ABABC, CBGA, CETS, BGWO), the repair-equipped variants and the
                   penalty variant
sets/              the set operators of set-based PSO, as pure functions
problems/          multidimensional knapsack, maximum diversity, feature selection
                   and set covering, with their instance data
experiments/       parallel runner, Sobol tuning, results database, statistics
paper-experiments/ one script per table or figure of the paper
tests/             test suite
run_tuning.py      tuning protocol (32 Sobol configurations per algorithm and domain)
run_experiment.py  experiment driver
main.py            small worked example
```

To add an algorithm, subclass `core.Optimizer` and implement `initialize`,
`step` and `get_best`; call the objective only through `self.evaluate`, which
counts evaluations, and draw all random numbers from `self.rng`.

## Data sources

The instance files are included for reproducibility and remain under the
terms of their sources:

- multidimensional knapsack: OR-Library (Beasley, 1990) suites mknap1, mknap2
  and the Chu–Beasley classes mknapcb1–9, and the Glover–Kochenberger
  instances gk01–gk11;
- maximum diversity: MDPLIB (Martí, Duarte, Martínez-Gavara and Sánchez-Oro);
- feature selection: UCI Machine Learning Repository datasets, Wine and WDBC
  as bundled with scikit-learn and the others cached from OpenML (Madelon,
  which the paper excludes, is fetched on demand);
- set covering: OR-Library.

## License

The code is released under the MIT License (`LICENSE`). The results in
`results/` and the database on Zenodo are released under CC BY 4.0.
