"""Extra experiments used in the technical report.

    python run_experiments.py sensitivity   # population size / mutation rate / selection pressure
    python run_experiments.py compare       # GA vs random search over independent seeds + uniqueness
    python run_experiments.py scenarios     # same engine, three different weight presets
    python run_experiments.py all
"""
import json, os, sys, time
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uav_ga import Config, build_environment, GeneticAlgorithm, RandomSearch
from uav_ga import visualize as viz

OUT = "results"
SEEDS = [1, 2, 3, 4]
INITIAL = dict(mutation_rate=0.08, tournament_k=3)   # settings before tuning (reference for the sweeps)


def make_env(cfg):
    return build_environment(cfg.grid_size, cfg.base, cfg.n_targets, cfg.env_seed)


def run_ga(cfg, seed, snaps=()):
    env = make_env(cfg)
    ga = GeneticAlgorithm(env, cfg, seed=seed)
    return env, ga, ga.run(snapshot_gens=snaps)


# ---------------------------------------------------------------- sensitivity
def sensitivity():
    base = replace(Config(generations=100, patience=None), **INITIAL).scaled()
    grid = {"population size": ("pop_size", [20, 50, 100, 200]),
            "mutation rate": ("mutation_rate", [0.01, 0.04, 0.08, 0.2, 0.4]),
            "tournament size (selection pressure)": ("tournament_k", [2, 3, 5, 10])}
    results, raw = {}, {}
    for label, (field, values) in grid.items():
        results[label], raw[label] = {}, {}
        for v in values:
            cfg = replace(base, **{field: v})
            fits, divs, covs, evals = [], [], [], []
            for s in SEEDS:
                _, _, r = run_ga(cfg, s)
                fits.append(r.best.fitness); divs.append(r.history["diversity"][-1])
                covs.append(r.best.info["coverage"]); evals.append(r.evaluations)
            results[label][v] = {"fit": fits, "div": divs}
            raw[label][str(v)] = dict(fit_mean=float(np.mean(fits)), fit_std=float(np.std(fits)),
                                      div_mean=float(np.mean(divs)), cov_mean=float(np.mean(covs)),
                                      evals=int(np.mean(evals)))
            print(label, v, raw[label][str(v)], flush=True)
    viz.plot_sensitivity(results, f"{OUT}/sensitivity.png")
    json.dump(raw, open(f"{OUT}/sensitivity.json", "w"), indent=1)


# -------------------------------------------------------------------- compare
def jaccard_dist(a, b):
    return 1.0 - (a & b).sum() / max(1, (a | b).sum())


def compare():
    cfg = replace(Config(patience=None), **INITIAL).scaled()
    env = make_env(cfg)
    lam = cfg.pen_end
    data = {"GA": [], "Random (sector-seeded)": [], "Random (uniform)": []}
    masks = {k: [] for k in data}
    rows = {k: [] for k in data}
    for s in SEEDS + [5]:
        ga = GeneticAlgorithm(env, cfg, seed=s); res = ga.run()
        budget = res.evaluations
        runs = {"GA": (res.best, ga.evaluator)}
        for name, strat in (("Random (sector-seeded)", "sector"), ("Random (uniform)", "uniform")):
            rs = RandomSearch(env, cfg, seed=s, strategy=strat)
            best, _ = rs.run(budget)
            runs[name] = (best, rs.evaluator)
        for name, (ind, ev) in runs.items():
            r = ev.evaluate(ind.genes, lam, return_maps=True)
            data[name].append(r["fitness"]); masks[name].append(r["qgrid"] > 0)
            rows[name].append({k: (bool(r[k]) if k == "feasible" else float(r[k]))
                               for k in ("fitness", "coverage", "detection", "energy", "time", "hazard", "feasible")})
        print("seed", s, {k: round(v[-1], 3) for k, v in data.items()}, flush=True)
    uniq = {}
    for name, ms in masks.items():
        d = [jaccard_dist(ms[i], ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms))]
        uniq[name] = float(np.mean(d))
    summary = {n: dict(fit_mean=float(np.mean(v)), fit_std=float(np.std(v)), fit_best=float(np.max(v)),
                       feasible_runs=int(sum(r["feasible"] for r in rows[n])), runs=len(v),
                       coverage=float(np.mean([r["coverage"] for r in rows[n]])),
                       detection=float(np.mean([r["detection"] for r in rows[n]])),
                       hazard=float(np.mean([r["hazard"] for r in rows[n]])),
                       energy=float(np.mean([r["energy"] for r in rows[n]])),
                       pairwise_jaccard_distance=uniq[n]) for n, v in data.items()}
    json.dump(dict(summary=summary, fitness=data, rows=rows), open(f"{OUT}/compare.json", "w"), indent=1)
    viz.plot_boxplot(data, f"{OUT}/boxplot_ga_vs_random.png")
    print(json.dumps(summary, indent=1))


# ------------------------------------------------------------------ scenarios
def scenarios():
    presets = {
        "Balanced (default)": {},
        "Coverage-first": dict(w_cov=0.50, w_det=0.30, w_energy=0.04, w_time=0.03, w_hazard=0.05),
        "Safety-first": dict(w_cov=0.25, w_det=0.20, w_energy=0.10, w_time=0.05, w_hazard=0.60),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    out = {}
    for ax, (name, kw) in zip(axes, presets.items()):
        cfg = replace(Config(patience=None), **{**INITIAL, **kw}).scaled()
        env, ga, res = run_ga(cfg, seed=11)
        r = ga.evaluator.evaluate(res.best.genes, cfg.pen_end, return_maps=True)
        viz.plot_paths(ax, env, res.best.genes, r["nodes"],
                       f"{name}\ncov={r['coverage']:.2f} det={r['detection']:.2f} "
                       f"haz={r['hazard']:.2f} E={r['energy']:.2f}")
        out[name] = {k: float(r[k]) for k in ("fitness", "coverage", "detection", "energy", "time", "hazard")}
        out[name]["feasible"] = bool(r["feasible"])
        print(name, out[name], flush=True)
    fig.tight_layout(); fig.savefig(f"{OUT}/weight_scenarios.png", dpi=150); plt.close(fig)
    json.dump(out, open(f"{OUT}/scenarios.json", "w"), indent=1)


# ---------------------------------------------------------------------- tuned
def tuned():
    """Five independent runs with the tuned defaults (mutation 0.02, tournament 5)."""
    cfg = replace(Config(patience=None), mutation_rate=0.02, tournament_k=5).scaled()
    env = make_env(cfg)
    rows = []
    for s in SEEDS + [5]:
        ga = GeneticAlgorithm(env, cfg, seed=s); r = ga.run()
        e = ga.evaluator.evaluate(r.best.genes, cfg.pen_end)
        rows.append({k: (bool(e[k]) if k == "feasible" else float(e[k]))
                     for k in ("fitness", "coverage", "detection", "energy", "time", "hazard", "feasible")})
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "feasible"}
    agg["fit_std"] = float(np.std([r["fitness"] for r in rows])); agg["fit_best"] = max(r["fitness"] for r in rows)
    agg["feasible_runs"] = int(sum(r["feasible"] for r in rows))
    json.dump(dict(config=dict(mutation_rate=0.02, tournament_k=5), summary=agg, rows=rows),
              open(f"{OUT}/tuned.json", "w"), indent=1)
    print(agg)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if mode in ("compare", "all"): compare()
    if mode in ("scenarios", "all"): scenarios()
    if mode in ("sensitivity", "all"): sensitivity()
    if mode in ("tuned", "all"): tuned()
    print(f"done in {time.time() - t0:.0f}s")
