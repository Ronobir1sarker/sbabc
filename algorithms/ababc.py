"""
ABABC — the adaptive binary artificial bee colony of Durgut & Aydın: the
most recent binary ABC among the competitors.

Sources, in order of authority:
  1. Durgut & Aydın (2021), "Adaptive binary artificial bee colony
     algorithm", Applied Soft Computing 101:107054 —
     defines the operator pool {binABC, disABC, ibinABC}, the reward
     (its eq. 17), credit assignment (eq. 18) and adaptive pursuit (eq. 14).
  2. Durgut & Aydın (2021), "Çok boyutlu sırt çantası problemi için adaptif
     ikili yapay arı kolonisi algoritması (AİYAK)", J. Fac. Eng. Arch. Gazi
     Univ. 36:4 2333-2348 — the MKP application this competitor
     reproduces: Chu & Beasley utility repair, and the tuned configuration
     SN=40, adaptive pursuit, average reward with window W=25, alpha=0.9,
     p_min=0.05, p_max=0.8, beta=0.5, limit=SN*D/2, 100k evaluations.
  3. Durgut (2020), "Improved binary artificial bee colony algorithm",
     arXiv:2003.11641 — exact ibinABC semantics (XOR truth table, q schedule
     with the tuned Q={0.3, 0.1}).
  4. Kashan et al. (2012), DisABC, Applied Soft Computing 12:342-352 — the
     dissimilarity operator and its linear phi schedule.
  5. The authors' own reference implementations
     (github.com/rafetdurgut: ibinABC MATLAB, RL-AOS-* Python) — used to
     resolve every formula the papers typeset ambiguously.

One cycle = employed bees, onlooker bees, credit/probability update, scout.
Per bee: an operator is drawn from the pool by roulette over the adaptive-
pursuit probabilities, a neighbour is drawn by fitness-proportional roulette
over the whole colony (the papers' pseudocode; topology is ignored, like
CBGA's panmictic population), the operator builds a candidate, the candidate
is repaired (Chu & Beasley DROP/ADD on MKP problems), evaluated (one counted
evaluation), rewarded, and greedily compared with the source (strict
improvement — the papers' rule, i.e. tie_break="current").

Documented interpretation choices where the sources disagree:
  - ibinABC bit count: both papers typeset eq. (11)/(16) ambiguously; the
    authors' Python (exp(-4t/tmax) * 0.1D outside the exponent) is the only
    version matching the prose ("more bits early, fewer late") and is used:
    d_t = rand()*0.1*D + exp(-4*t/tmax)*0.1*D + 1.
  - ibinABC q rule: the papers' prose ("neighbor better => copy its bits
    exactly, q=0") is followed; the authors' SUKP code inverts the condition
    because its fitness formula is minimization-oriented on a maximization
    problem — a port bug we do not replicate. Comparisons use Deb's rule, so
    the operator stays constraint- and direction-correct.
  - Reward window: rewards (0 on failure) are recorded per operator
    *application* and the window aggregates the last `reward_window` of them
    (the authors' code); credits and pursuit probabilities update once per
    cycle (the papers' update phase).
  - Reward scale: eq. 17's f(v)/gbest factor, computed on objective values;
    non-finite values (possible only without repair) yield reward 0.
  - Roulette fitness: raw value-proportional probabilities (the authors'
    code; near-uniform on MKP's narrow value bands). Karaboga's 1/(1+f)
    fitness is minimization-oriented and would invert preferences on MKP;
    the framework's min-shifted roulette applies too much pressure — the A/B
    reproduction runs show it drops weing3 from the paper's 100% success to
    ~77%. Falls back to the constraint-aware framework rule whenever values
    are unusable as weights (infeasible or non-positive).
  - disABC phi schedule: linear phiMax -> phiMin over the run; defaults
    0.9 -> 0.5, the setting of Kashan's paper and the authors' MATLAB
    contemporary with AİYAK. The integer model is solved exactly in closed
    form per j (nearest k), not by their brute-force double loop.
  - Utility weights for repair: AİYAK's eq. 21 leaves the surrogate weights
    w_i unspecified; the default is w_i = 1 ("unit"), which reproduced the
    paper's results at least as well as Chu & Beasley's LP-dual scheme
    ("lp_dual", shared with algorithms/cbga.py) in the A/B reproduction runs.

Reproduction status (30 runs x 100k evaluations on the 8 Weing instances of
the paper's Table 3, rows 31-38): success rates match the paper on 7 of 8
instances (100% each; weing3 required the raw-selection choice above). On
weing7 (2x105, optimum 1095445) this implementation reaches the optimum in
~0-3% of runs vs the paper's 40%, plateauing at 1095382-1095384 — exactly
the level the paper reports for its OTHER competitors (BFOA/HHS 1095384) —
because adaptive pursuit concentrates on disABC late in the run. Six
documented variant hypotheses (d_t forms, selection pressure, utility
weights, phi_min, inverted q, tournament neighbours) were A/B-tested without
recovering the paper's weing7 rate; the authors' exact AİYAK source is not
public. Disclose this residual gap wherever ABABC results are reported.

`limit=None` (default) applies the paper's rule limit = SN*D/2, resolved per
problem in set_problem — the stored parameter value None documents the rule.
`schedule_iterations` anchors the ibinABC/disABC schedules (the paper derives
t_max = budget / (2*SN); default 1250 = 100k evaluations at SN=40) — set it
to the campaign's planned iteration count.

On problems without MKP extras the repair is skipped (candidates may then be
infeasible; Deb's rule and the reward guard handle them), so the algorithm
still runs on other binary domains, with that deviation disclosed.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from algorithms.artificial_bee_colony import ArtificialBeeColony
from algorithms.cbga import chu_beasley_repair, mkp_arrays, pseudo_utility_orders


class AdaptiveBinaryABC(ArtificialBeeColony):

    name = "ABABC"

    OPERATORS = ("binABC", "disABC", "ibinABC")

    def __init__(
        self,
        num_food_sources: int = 40,
        limit: int | None = None,  # None: the paper's limit = SN*D/2, per problem
        credit_alpha: float = 0.9,  # eq. 18 adaptation factor
        pursuit_beta: float = 0.5,  # eq. 14 learning coefficient
        p_min: float = 0.05,
        p_max: float = 0.8,
        reward_type: str = "average",  # "instant" | "average" | "extreme"
        reward_window: int = 25,  # W, in operator applications
        q_max: float = 0.3,  # ibinABC complement-probability schedule
        q_min: float = 0.1,
        phi_max: float = 0.9,  # disABC dissimilarity schedule
        phi_min: float = 0.5,
        schedule_iterations: int = 1250,  # t_max for the operator schedules
        operators: tuple = OPERATORS,  # subset for ablations
        utility_weights: str = "unit",  # repair surrogate weights: "unit" | "lp_dual"
        tie_break: str = "current",  # papers use strict improvement
        **topology_kwargs,
    ):
        super().__init__(
            num_food_sources=num_food_sources,
            limit=0 if limit is None else limit,
            tie_break=tie_break,
            **topology_kwargs,
        )
        if reward_type not in ("instant", "average", "extreme"):
            raise ValueError(f"reward_type must be instant/average/extreme, got {reward_type!r}")
        unknown = set(operators) - set(self.OPERATORS)
        if unknown or not operators:
            raise ValueError(f"operators must be a non-empty subset of {self.OPERATORS}, got {operators}")
        if not 0 <= p_min <= p_max <= 1:
            raise ValueError(f"need 0 <= p_min <= p_max <= 1, got {p_min}, {p_max}")
        if schedule_iterations <= 0:
            raise ValueError(f"schedule_iterations must be positive, got {schedule_iterations}")
        if utility_weights not in ("unit", "lp_dual"):
            raise ValueError(f"utility_weights must be 'unit' or 'lp_dual', got {utility_weights!r}")

        self.limit = limit  # None is recorded in the DB; resolved in set_problem
        self._auto_limit = limit is None
        self._limit = limit  # the per-problem resolved value (set_problem)
        self.credit_alpha = credit_alpha
        self.pursuit_beta = pursuit_beta
        self.p_min = p_min
        self.p_max = p_max
        self.reward_type = reward_type
        self.reward_window = reward_window
        self.q_max = q_max
        self.q_min = q_min
        self.phi_max = phi_max
        self.phi_min = phi_min
        self.schedule_iterations = schedule_iterations
        self.operators = tuple(operators)
        self.utility_weights = utility_weights

    # ─────────────────────── problem-specific setup ────────────────────

    def set_problem(self, problem):
        super().set_problem(problem)
        if problem is None:
            return
        # resolved into a separate attribute: self.limit stays exactly the
        # constructor value (None = the SN*D/2 rule) so the parameters the
        # store records never drift after a sequential in-process run
        self._limit = (
            self.num_food_sources * self.dim // 2 if self._auto_limit else self.limit
        )
        try:
            profits, weights, capacity = mkp_arrays(problem)
        except ValueError:
            self._repair_data = None  # non-MKP domain: run without repair
        else:
            if self.utility_weights == "lp_dual":
                orders = pseudo_utility_orders(profits, weights, capacity)
            else:
                # AİYAK eq. 21 with unit surrogate weights: u_j = p_j / sum_i r_ij
                weight_sums = weights.sum(axis=0)
                with np.errstate(divide="ignore"):
                    utility = np.where(weight_sums > 0, profits / weight_sums, np.inf)
                drop = np.argsort(utility, kind="stable")
                orders = (drop, drop[::-1])
            self._repair_data = (weights, capacity, *orders)

    def _repair(self, x):
        if self._repair_data is not None:
            chu_beasley_repair(x, *self._repair_data)

    def random_food_sources(self, n):
        """Random sources, repaired to feasibility (the paper's 'valid random
        solutions') — used by initialize() and the scout phase alike."""
        sources = super().random_food_sources(n)
        for row in sources:
            self._repair(row)
        return sources

    # ─────────────────────── Optimizer interface ───────────────────────

    def initialize(self, initial_population=None):
        K = len(self.operators)
        self._op_probs = np.full(K, 1.0 / K)  # p_{i,0} = 1/K (eq. 14)
        self._op_credits = np.zeros(K)
        self._op_rewards = [deque(maxlen=self.reward_window) for _ in range(K)]
        self._iteration = 0
        super().initialize(initial_population)

    def step(self, iteration):
        self._iteration = iteration
        self._employed_bee_phase()
        self._onlooker_bee_phase()
        self._update_operator_selection()  # the papers' update phase
        self._scout_bee_phase()
        self._remember_best()

    def _scout_bee_phase(self):
        # the papers replace only the FIRST source exceeding the limit per
        # cycle, not all of them like the framework's base ABC
        for n in range(self.num_food_sources):
            if self.trials[n] > self._limit:
                self._reinitialize_food_source(n)
                break

    # ───────────────────── neighbour + operator choice ─────────────────

    def _selection_probabilities(self):
        """Raw value-proportional roulette (the authors' implementation).
        On MKP's narrow value bands this is nearly uniform — much weaker
        selection pressure than the framework's min-shifted roulette, and the
        A/B reproduction runs show the paper's results need it (the shifted
        roulette drops weing3 from 100% to ~77% success). Falls back to the
        constraint-aware parent rule when values are unusable as weights."""
        values = np.asarray(self.food_values, dtype=np.float64)
        if (
            self.optimization_type == "max"
            and np.all(np.isfinite(values))
            and np.all(values > 0)
            and np.all(self.food_violations <= 0)
        ):
            return values / values.sum()
        return super()._selection_probabilities()

    def random_neighbor(self, n: int) -> int:
        """Fitness-proportional roulette over the whole colony (the papers'
        pseudocode), resampled until it differs from the source itself."""
        probs = self._selection_probabilities()
        neighbor = int(self.rng.choice(self.num_food_sources, p=probs))
        while neighbor == n:
            neighbor = int(self.rng.choice(self.num_food_sources, p=probs))
        return neighbor

    def _select_operator(self) -> int:
        probs = self._op_probs / self._op_probs.sum()
        return int(self.rng.choice(len(self.operators), p=probs))

    def distort_operator(self, source, neighbor):
        op = self._select_operator()
        candidate = getattr(self, f"_op_{self.operators[op]}")(source, neighbor)
        self._repair(candidate)
        value, violation = self.evaluate_food_source(candidate)
        self._record_reward(op, value, violation, source)
        return candidate, value, violation

    def _progress(self) -> float:
        return min(self._iteration / self.schedule_iterations, 1.0)

    # ─────────────────────────── the operator pool ─────────────────────

    def _op_binABC(self, source, neighbor):
        """Kiran & Gündüz's XOR operator on one random bit: the candidate bit
        becomes the neighbour's bit or its complement, 50/50."""
        candidate = self.food_sources[source].copy()
        j = int(self.rng.integers(self.dim))
        neighbor_bit = self.food_sources[neighbor, j]
        candidate[j] = neighbor_bit if self.rng.random() < 0.5 else 1 - neighbor_bit
        return candidate

    def _op_ibinABC(self, source, neighbor):
        """Durgut's improved binABC: d_t bits (many early, few late); each
        processed bit copies the neighbour, or takes its complement with
        probability q — and q = 0 whenever the neighbour is the better source,
        so good neighbours are copied verbatim."""
        progress = self._progress()
        d_t = int(
            self.rng.random() * 0.1 * self.dim
            + np.exp(-4.0 * progress) * 0.1 * self.dim
            + 1
        )
        d_t = min(d_t, self.dim)
        q = self.q_max - (self.q_max - self.q_min) * progress
        if self.is_first_better(
            self.food_values[neighbor], self.food_values[source],
            self.food_violations[neighbor], self.food_violations[source],
        ):
            q = 0.0
        dims = self.rng.permutation(self.dim)[:d_t]
        candidate = self.food_sources[source].copy()
        complement = self.rng.random(d_t) < q
        neighbor_bits = self.food_sources[neighbor, dims]
        candidate[dims] = np.where(complement, 1 - neighbor_bits, neighbor_bits)
        return candidate

    def _op_disABC(self, source, neighbor):
        """Kashan's dissimilarity operator: build a candidate whose Jaccard
        dissimilarity to the source approximates phi * dissimilarity(source,
        neighbour), by keeping n1 of the source's ones and adding n0 of its
        zeros. The integer model min |1 - k/(m1+j) - A| is solved exactly:
        for each j the optimal k is the nearest integer to (1-A)*(m1+j),
        within the authors' reference search ranges (see inline note)."""
        x = self.food_sources[source]
        nb = self.food_sources[neighbor]
        phi = self.phi_max - (self.phi_max - self.phi_min) * self._progress()

        ones = np.flatnonzero(x == 1)
        zeros = np.flatnonzero(x == 0)
        m1, m0 = len(ones), len(zeros)

        n1_keep = n0_add = 0
        if m1 and m0:
            m11 = int(np.sum((x == 1) & (nb == 1)))
            union = int(np.sum((x == 1) | (nb == 1)))  # >= m1 >= 1 here
            target_dissim = phi * (1.0 - m11 / union)
            # the search ranges k in [0, m1-1], j in [0, m0-1] mirror the
            # authors' reference implementation (which never keeps ALL ones
            # nor adds ALL zeros); Kashan's model itself would also admit
            # k = m1 / j = m0. Within those ranges the model is solved
            # exactly: per j, the k nearest to (1-A)(m1+j) is optimal.
            j = np.arange(m0)
            k = np.clip(np.rint((1.0 - target_dissim) * (m1 + j)), 0, m1 - 1)
            error = np.abs(1.0 - k / (m1 + j) - target_dissim)
            best = int(np.argmin(error))
            n1_keep, n0_add = int(k[best]), int(j[best])

        candidate = np.zeros_like(x)
        candidate[ones[self.rng.permutation(m1)[:n1_keep]]] = 1
        candidate[zeros[self.rng.permutation(m0)[:n0_add]]] = 1
        return candidate

    # ─────────────── credit assignment + adaptive pursuit ──────────────

    def _record_reward(self, op, value, violation, source):
        """Eq. 17: r = (f(candidate)/f(gbest)) * (f(candidate) - f(source)),
        zero unless the candidate strictly improves on its source (Deb)."""
        reward = 0.0
        if self.is_first_better(
            value, self.food_values[source], violation, self.food_violations[source]
        ):
            current = self.food_values[source]
            improvement = value - current if self.optimization_type == "max" else current - value
            scale = 1.0
            if np.isfinite(self.best_value) and self.best_value != 0 and np.isfinite(value):
                scale = abs(value) / abs(self.best_value)
            reward = scale * improvement
            if not np.isfinite(reward) or reward < 0:
                reward = 0.0
        self._op_rewards[op].append(reward)

    def _aggregate_reward(self, op) -> float:
        window = self._op_rewards[op]
        if not window:
            return 0.0
        if self.reward_type == "instant":
            return window[-1]
        if self.reward_type == "average":
            return float(np.mean(window))
        return float(np.max(window))  # extreme

    def _update_operator_selection(self):
        """The papers' update phase, once per cycle: credit eq. 18, then the
        adaptive-pursuit probability update eq. 14."""
        K = len(self.operators)
        rewards = np.array([self._aggregate_reward(op) for op in range(K)])
        self._op_credits = (1 - self.credit_alpha) * self._op_credits + self.credit_alpha * rewards

        if np.ptp(self._op_credits) == 0:
            return  # all credits equal (e.g. at the start): stay uniform
        winners = np.flatnonzero(self._op_credits == self._op_credits.max())
        best = int(winners[0]) if len(winners) == 1 else int(self.rng.choice(winners))
        for op in range(K):
            target = self.p_max if op == best else self.p_min
            self._op_probs[op] += self.pursuit_beta * (target - self._op_probs[op])
