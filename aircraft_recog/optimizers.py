"""Step 7 + feature 1: nature-inspired hyper-parameter search.

All four searchers maximise a fitness function (validation accuracy after a
short training run) over the same search space and get the SAME number of
fitness evaluations, so the comparison is fair.

    * Random search                   (baseline)
    * Particle Swarm Optimization     bird flocks          [Kennedy & Eberhart 1995]
    * Genetic Algorithm               natural evolution    [Holland 1975]
    * Grey Wolf Optimizer             wolf-pack hunting    [Mirjalili et al. 2014]

The code is written out in full (about 40 lines per method) so it can be
explained in the viva, instead of calling a black-box library.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------- search space
# Each particle / chromosome / wolf is a vector of 4 real numbers:
#   [log10(learning rate), dropout, batch-size index, number of unfrozen blocks]
LOW = np.array([-4.0, 0.1, 0.0, 1.0])
HIGH = np.array([-2.0, 0.6, 2.999, 4.999])
BATCHES = [16, 32, 64]


def decode(v) -> dict:
    v = np.clip(np.asarray(v, float), LOW, HIGH)
    return {
        "lr": float(10 ** v[0]),
        "dropout": float(round(v[1], 3)),
        "batch": BATCHES[int(v[2])],
        "unfrozen": int(v[3]),
    }


def _key(hp: dict):
    return (round(np.log10(hp["lr"]), 2), round(hp["dropout"], 2), hp["batch"], hp["unfrozen"])


@dataclass
class SearchResult:
    method: str
    best_hp: dict
    best_score: float
    evaluations: list = field(default_factory=list)  # (index, hp, score)
    curve: list = field(default_factory=list)        # best-so-far after each round
    runs_used: int = 0
    unique_runs: int = 0
    time_s: float = 0.0

    def to_dict(self):
        return dict(method=self.method, best_hp=self.best_hp, best_score=self.best_score,
                    curve=self.curve, runs_used=self.runs_used, unique_runs=self.unique_runs,
                    time_s=self.time_s,
                    evaluations=[dict(i=i, hp=h, score=s) for i, h, s in self.evaluations])


class _Evaluator:
    """Wraps the fitness function: counts calls, caches repeats, tracks the best."""

    def __init__(self, fitness, method, verbose=True):
        self.fitness, self.method, self.verbose = fitness, method, verbose
        self.cache, self.evals = {}, []
        self.best_score, self.best_hp = -np.inf, None
        self.t0 = time.time()

    def __call__(self, v) -> float:
        hp = decode(v)
        k = _key(hp)
        if k not in self.cache:
            self.cache[k] = float(self.fitness(hp))
        s = self.cache[k]
        self.evals.append((len(self.evals) + 1, hp, s))
        if s > self.best_score:
            self.best_score, self.best_hp = s, hp
        if self.verbose:
            print(f"[{self.method}] run {len(self.evals):>2}: {hp} -> {s:.4f} (best {self.best_score:.4f})")
        return s

    def result(self, curve):
        return SearchResult(self.method, self.best_hp, self.best_score, self.evals, curve,
                            len(self.evals), len(self.cache), time.time() - self.t0)


def _rand(rng, n):
    return LOW + rng.rand(n, len(LOW)) * (HIGH - LOW)


# ============================================================== random search
def random_search(fitness, budget=30, pop=6, seed=0, verbose=True) -> SearchResult:
    rng = np.random.RandomState(seed)
    ev = _Evaluator(fitness, "Random", verbose)
    curve = []
    for i, v in enumerate(_rand(rng, budget)):
        ev(v)
        if (i + 1) % pop == 0 or i + 1 == budget:
            curve.append(ev.best_score)
    return ev.result(curve)


# ================================================== Particle Swarm Optimization
def pso(fitness, budget=30, pop=6, seed=0, w=0.7, c1=1.5, c2=1.5, verbose=True) -> SearchResult:
    """v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x);  x = x + v"""
    rng = np.random.RandomState(seed)
    ev = _Evaluator(fitness, "PSO", verbose)
    rounds = max(1, budget // pop)
    span = HIGH - LOW
    x = _rand(rng, pop)                                  # positions
    v = (rng.rand(pop, len(LOW)) - 0.5) * 0.2 * span     # velocities
    vmax = 0.3 * span
    score = np.array([ev(p) for p in x])
    pbest, pbest_s = x.copy(), score.copy()
    g = int(np.argmax(pbest_s))
    gbest, gbest_s = pbest[g].copy(), pbest_s[g]
    curve = [gbest_s]
    for _ in range(rounds - 1):
        r1, r2 = rng.rand(pop, len(LOW)), rng.rand(pop, len(LOW))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, LOW, HIGH)
        score = np.array([ev(p) for p in x])
        better = score > pbest_s
        pbest[better], pbest_s[better] = x[better], score[better]
        g = int(np.argmax(pbest_s))
        if pbest_s[g] > gbest_s:
            gbest, gbest_s = pbest[g].copy(), pbest_s[g]
        curve.append(gbest_s)
    return ev.result(curve)


# ========================================================== Genetic Algorithm
def genetic_algorithm(fitness, budget=30, pop=6, seed=0, mutation_rate=0.25,
                      tournament=3, verbose=True) -> SearchResult:
    """Selection (tournament) -> crossover (blend) -> mutation (Gaussian), with elitism."""
    rng = np.random.RandomState(seed)
    ev = _Evaluator(fitness, "GA", verbose)
    gens = max(1, budget // pop)
    span = HIGH - LOW
    popu = _rand(rng, pop)
    fit = np.array([ev(c) for c in popu])
    curve = [fit.max()]

    def select():
        idx = rng.choice(pop, size=min(tournament, pop), replace=False)
        return popu[idx[np.argmax(fit[idx])]]

    for _ in range(gens - 1):
        elite, elite_f = popu[np.argmax(fit)].copy(), fit.max()
        children = []
        while len(children) < pop:
            a, b = select(), select()
            alpha = rng.uniform(-0.25, 1.25, size=len(LOW))    # BLX-style blend crossover
            child = a + alpha * (b - a)
            mut = rng.rand(len(LOW)) < mutation_rate
            child = child + mut * rng.normal(0, 0.15, len(LOW)) * span
            children.append(np.clip(child, LOW, HIGH))
        children = np.array(children)
        child_fit = np.array([ev(c) for c in children])
        # elitism: the best parent survives in place of the worst child (no extra run)
        worst = int(np.argmin(child_fit))
        if elite_f > child_fit[worst]:
            children[worst], child_fit[worst] = elite, elite_f
        popu, fit = children, child_fit
        curve.append(max(curve[-1], fit.max()))
    return ev.result(curve)


# ========================================================= Grey Wolf Optimizer
def grey_wolf(fitness, budget=30, pop=6, seed=0, verbose=True) -> SearchResult:
    """The three best wolves (alpha, beta, delta) lead; the pack moves around them.
    The coefficient `a` falls from 2 to 0: exploration first, then exploitation."""
    rng = np.random.RandomState(seed)
    ev = _Evaluator(fitness, "GWO", verbose)
    iters = max(1, budget // pop)
    wolves = _rand(rng, pop)
    fit = np.array([ev(w_) for w_ in wolves])
    curve = [fit.max()]
    leaders = lambda: [wolves[i].copy() for i in np.argsort(-fit)[:3]]
    alpha, beta, delta = (leaders() + [wolves[0]] * 3)[:3]
    for t in range(1, iters):
        a = 2 - 2 * t / max(1, iters - 1)
        new = np.empty_like(wolves)
        for i in range(pop):
            moves = []
            for leader in (alpha, beta, delta):
                A = 2 * a * rng.rand(len(LOW)) - a
                C = 2 * rng.rand(len(LOW))
                D = np.abs(C * leader - wolves[i])
                moves.append(leader - A * D)
            new[i] = np.clip(np.mean(moves, axis=0), LOW, HIGH)
        new_fit = np.array([ev(w_) for w_ in new])
        # keep the better of old and new position for each wolf (greedy)
        improve = new_fit > fit
        wolves[improve], fit[improve] = new[improve], new_fit[improve]
        alpha, beta, delta = (leaders() + [wolves[0]] * 3)[:3]
        curve.append(fit.max())
    return ev.result(curve)


METHODS = {
    "Random search": random_search,
    "Particle Swarm Optimization": pso,
    "Genetic Algorithm": genetic_algorithm,
    "Grey Wolf Optimizer": grey_wolf,
}
