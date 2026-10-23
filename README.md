# UAV Fleet Strategy Engine (Genetic Algorithm)

Artificial Intelligence – Assignment 2 (CLO 4), BSCS-5, Bahria University Lahore Campus.

A Genetic Algorithm (GA) that plans a **3-drone fleet** over a discretised **50 × 50 disaster grid**:
waypoint trajectories, speed adjustments and sensor power/activation schedules are evolved together
so that the fleet covers high-priority zones and finds targets while spending little energy and time
and avoiding hazard pockets (wind / radiation) and no-fly zones.

## Project structure

```
uav-ga-strategy-engine/
├── run_simulation.py      # main entry point: one GA run + random-search baselines + all plots
├── run_experiments.py     # sensitivity analysis, GA-vs-random study, weight scenarios
├── requirements.txt
├── uav_ga/
│   ├── config.py          # every tunable parameter (dataclass)
│   ├── environment.py     # grid: priority map, hazard map, no-fly zones, hidden targets
│   ├── chromosome.py      # hybrid real/binary encoding + repair operator
│   ├── fitness.py         # multi-objective fitness + constraint penalty
│   ├── operators.py       # selection, 3 crossovers, spatial mutation
│   ├── ga.py              # GeneticAlgorithm class (population loop, termination, diversity control)
│   ├── baseline.py        # random-search baselines (equal evaluation budget)
│   └── visualize.py       # matplotlib figures
├── tests/test_smoke.py    # quick sanity tests
└── results/               # generated figures + JSON metrics
```

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/<your-username>/uav-ga-strategy-engine.git
cd uav-ga-strategy-engine
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

```bash
# 1) Main simulation (~1 min): GA run, random-search baselines, convergence + path + coverage plots
python run_simulation.py                      # defaults: 50x50 grid, population 100, 200 generations
python run_simulation.py --size 100 --pop 150 --gens 300 --seed 7 --out results_100

# 2) Extra experiments used in the report (~15 min in total)
python run_experiments.py compare             # GA vs random search over 5 seeds + solution uniqueness
python run_experiments.py scenarios           # three weight presets (balanced / coverage-first / safety-first)
python run_experiments.py sensitivity         # population size, mutation rate, selection pressure
python run_experiments.py tuned               # 5 runs with the tuned default hyper-parameters

# 3) Tests
python -m pytest -q
```

Runs are reproducible: the world is fixed by `env_seed` and the GA by `--seed`.

## Design summary

| Component | Implementation |
|---|---|
| **Chromosome** | `genes[drone, waypoint] = [x, y, speed, sensor power, sensor-active bit]` – real-valued waypoints/speed/power plus a binary activation flag (hybrid encoding, 3 × 10 × 5 genes). Routes start and end at the base station. |
| **Fitness** | `F = 0.35·Coverage + 0.25·Detection − 0.12·Energy − 0.08·Time − 0.20·Hazard − λ·Violation` (all terms normalised). |
| **Hard constraints** | Range limit, battery limit, no-fly zones. *Repair* (clip to grid, snap binary genes, push waypoints out of no-fly cells) + *dynamic penalty* λ growing from 2 to 6 over the run. |
| **Selection** | k-tournament (k = 5) with elitism (2). |
| **Crossover** | Route swap (whole drone routes), segment swap (two-point), geometric/blend crossover on coordinates – chosen at random per mating. |
| **Mutation** | Annealed Gaussian shift, hazard-aware escape (moves away from the hazard gradient), random relocation, sensor/speed perturbation, route swap and segment inversion (2-opt style). |
| **Diversity control** | Population diversity is monitored; mutation rate is boosted ×2.5 when diversity collapses and random/sector-seeded immigrants are injected on stagnation. |
| **Termination** | Maximum generations, or no improvement for 60 generations (after the penalty ramp has finished). |

## Results (50 × 50 grid, fixed map, 200 generations, population 100)

| Method (mean of 5 independent seeds, equal budget ≈ 20 000 evaluations) | Fitness | Coverage | Detection | Feasible runs |
|---|---|---|---|---|
| **GA (tuned: mutation 0.02, tournament 5)** | **0.370 ± 0.006** | 0.95 | 0.76 | 5 / 5 |
| GA (initial: mutation 0.08, tournament 3) | 0.288 ± 0.005 | 0.89 | 0.64 | 5 / 5 |
| Random search (sector-seeded) | 0.167 ± 0.005 | 0.76 | 0.52 | 5 / 5 |
| Random search (uniform) | −2.74 ± 0.62 | 0.69 | 0.46 | 0 / 5 |

Figures produced in `results/`:

| File | Content |
|---|---|
| `convergence.png` | Best vs. average fitness per generation, population diversity, feasible fraction |
| `evolution_paths_coverage.png` | Flight paths and coverage heat-maps at generations 0, 10, 40 and the final generation |
| `final_solution.png` | Final evolved plan with detected / missed targets |
| `ga_vs_random.png`, `boxplot_ga_vs_random.png` | GA vs random search |
| `sensitivity.png` | Hyper-parameter sensitivity |
| `weight_scenarios.png` | Different weightings give different, equally valid plans |

The 3–5 page technical report is in `report/Technical_Report.docx`.

## Notes and limitations

* All results are for one fixed synthetic map; other maps may need re-tuning.
* The weighted-sum fitness collapses the objectives into one number; different weights (see
  `weight_scenarios.png`) give different trade-off solutions.
