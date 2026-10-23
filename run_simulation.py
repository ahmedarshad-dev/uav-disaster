"""Run one full GA simulation, a random-search baseline, and save plots to ./results."""
import argparse, json, os
import numpy as np
from uav_ga import Config, build_environment, GeneticAlgorithm, RandomSearch
from uav_ga import visualize as viz


def main():
    ap = argparse.ArgumentParser(description="UAV fleet Genetic Algorithm strategy engine")
    ap.add_argument("--size", type=int, default=50, help="grid size (50 or 100)")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = Config(grid_size=a.size, pop_size=a.pop, generations=a.gens).scaled()
    env = build_environment(cfg.grid_size, cfg.base, cfg.n_targets, cfg.env_seed)
    ga = GeneticAlgorithm(env, cfg, seed=a.seed)
    snaps = (0, max(1, a.gens // 20), max(2, a.gens // 5))
    res = ga.run(snapshot_gens=snaps, verbose=True)
    lam = cfg.pen_end
    print(f"\nGA finished: {res.generations_run} generations, {res.evaluations} evaluations ({res.stop_reason})")

    # ---- plots ----
    viz.plot_convergence(res.history, f"{a.out}/convergence.png")
    viz.plot_evolution(env, ga.evaluator, res.snapshots, f"{a.out}/evolution_paths_coverage.png", lam)
    final = viz.plot_final(env, ga.evaluator, res.best.genes, f"{a.out}/final_solution.png", lam)

    # ---- baselines with the same evaluation budget ----
    runs = []
    for name, strat, col in (("Random search (uniform)", "uniform", "#7f7f7f"),
                             ("Random search (sector-seeded)", "sector", "#ff7f0e")):
        best, curve = RandomSearch(env, cfg, seed=a.seed, strategy=strat).run(res.evaluations)
        runs.append((name, (best, curve), col))
        print(f"{name}: fitness {best.fitness:.4f} feasible={best.info['feasible']} coverage={best.info['coverage']:.3f}")
    viz.plot_vs_random(env, ga.evaluator, res, runs, f"{a.out}/ga_vs_random.png", lam)

    summary = dict(config=cfg.__dict__, generations=res.generations_run, evaluations=res.evaluations,
                   stop_reason=res.stop_reason,
                   ga_best={k: (float(v) if not isinstance(v, bool) else v) for k, v in final.items()
                            if k not in ("qgrid", "nodes")},
                   random_search={n: dict(fitness=float(b.fitness), feasible=bool(b.info["feasible"]),
                                          coverage=float(b.info["coverage"]), detection=float(b.info["detection"]))
                                  for n, (b, _), _ in runs},
                   history=res.history)
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    np.save(f"{a.out}/best_chromosome.npy", res.best.genes)
    print("Saved plots, summary.json and best_chromosome.npy to", a.out)


if __name__ == "__main__":
    main()
