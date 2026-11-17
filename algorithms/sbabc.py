"""
Set-Based Artificial Bee Colony (SBABC), the algorithm proposed in the paper.

It keeps the ABC skeleton (employed/onlooker/scout phases, trial counters,
Deb's rule) and replaces the numeric perturbation with the set-based
operators of Langeveld & Engelbrecht's SBPSO (see sets/operators.py):
food sources are subsets of items, and a candidate is built by moving
toward/away from a neighbor, dropping shared elements, and adding rare
elements via k-tournament on marginal objective values.

Distort operator versions (hand-tuned settings that worked well on the
small MKP benchmarks, carried over from the original experiments):
  1  towards-or-away          beta1=1 (main run) or beta1=0.1 with von Neumann topology
  3  + drop-common, add-rare  beta1=0.5, beta2=1,   beta3=3, k=11 (or k=3), limit=120
  4  add-rare on both branches beta1=0.5, beta2=1.5, beta3=3, k=5, limit=120
  5  spec-driven               the ablation instrument: candidate_specs names each
                               candidate as a subset of {m: move toward, d: drop
                               common, a: add rare}; the default
                               ("m", "md", "ma", "mda") reproduces version 4
                               bit-identically. shared_add_velocity=True instead
                               builds ONE addition velocity from the current
                               source's marginals and reuses it across all "a"
                               candidates — the manuscript's V3 semantics.

The candidate_set alias ("four" | "single") names the two candidate_specs
layouts the experiments compare, so the Sobol tuner can treat the
operator layout as an ordinary categorical parameter (tuples cannot ride
through its choice-list machinery — np.array(choices, dtype=object) would
collapse equal-length tuples into a 2-D array). The layout is a tuned
parameter, chosen per domain by run_tuning.py (results/tuning/*.json):
"single" on MKP, MDP and SCP, "four" on feature selection. "single" spends
one evaluation per bee where "four" spends four, which the ablation showed
buys ~3.2x the iterations under an FE-equal budget and wins on constrained
problems. The class default ("four") is kept for backward reproducibility;
the paper's runs never rely on it. An explicit candidate_specs always wins
over the alias — the ablation arms rely on that.

(A former version 2 was removed: it referenced undefined variables and a
nonexistent objective function, so it had never actually run.)
"""

from __future__ import annotations

import numpy as np

from algorithms.artificial_bee_colony import ArtificialBeeColony
from sets import operators as ops


class SetBasedABC(ArtificialBeeColony):

    name = "SBABC"

    DISTORT_OPERATOR_VERSIONS = {
        1: "_distort_towards_or_away",
        3: "_distort_drop_common_add_rare",
        4: "_distort_add_rare_on_both",
        5: "_distort_from_specs",
    }

    CANDIDATE_SETS = {
        "four": ("m", "md", "ma", "mda"),
        "single": ("mda",),
    }

    def __init__(
        self,
        beta1: float = 0.5,  # attraction toward / away from the neighbor, in [0, 1]
        beta2: float = 1.5,  # how many elements shared with the neighbor to drop
        beta3: float = 3.0,  # how many rare elements to try to add
        k: int = 5,  # tournament size when adding elements
        distort_operator_version: int = 4,
        scout_bee_phase_version: int = 1,
        candidate_specs: tuple | None = None,  # version 5 only; None -> from candidate_set
        candidate_set: str = "four",  # tuner-friendly alias; explicit candidate_specs wins
        shared_add_velocity: bool = False,  # version 5 only: one V3 from the source's marginals
        **abc_kwargs,
    ):
        super().__init__(**abc_kwargs)
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError(
                f"beta1 must be in [0, 1] (it scales a velocity; beyond 1, "
                f"scalar_mul is undefined and would crash mid-run only on the "
                f"random iteration where r1*beta1 > 1 is first drawn), got {beta1}"
            )
        if beta2 < 0 or beta3 < 0:
            raise ValueError(f"beta2/beta3 must be >= 0, got beta2={beta2}, beta3={beta3}")
        if int(k) < 1:
            raise ValueError(f"k must be a positive tournament size, got {k}")
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.k = int(k)

        if distort_operator_version not in self.DISTORT_OPERATOR_VERSIONS:
            raise ValueError(
                f"distort_operator_version must be one of {sorted(self.DISTORT_OPERATOR_VERSIONS)}, "
                f"got {distort_operator_version}"
            )
        if scout_bee_phase_version not in (1, 2):
            raise ValueError(f"scout_bee_phase_version must be 1 or 2, got {scout_bee_phase_version}")
        self.distort_operator_version = distort_operator_version
        self.scout_bee_phase_version = scout_bee_phase_version
        self._check_scout_v2_premise()  # catches an explicit "min" already here

        if candidate_set not in self.CANDIDATE_SETS:
            raise ValueError(
                f"candidate_set must be one of {sorted(self.CANDIDATE_SETS)}, got {candidate_set!r}"
            )
        self.candidate_set = candidate_set
        if candidate_specs is None:
            candidate_specs = self.CANDIDATE_SETS[candidate_set]
        candidate_specs = tuple(candidate_specs)
        for spec in candidate_specs:
            if not spec or not set(spec) <= set("mda") or len(set(spec)) != len(spec):
                raise ValueError(
                    f"Each candidate spec must be a non-empty combination of 'm', 'd', 'a' "
                    f"without repeats, got {spec!r}"
                )
        if not candidate_specs:
            raise ValueError("candidate_specs must name at least one candidate")
        self.candidate_specs = candidate_specs
        self.shared_add_velocity = bool(shared_add_velocity)

    def _check_scout_v2_premise(self):
        if self.scout_bee_phase_version == 2 and self.optimization_type == "min":
            raise ValueError(
                "scout_bee_phase_version=2 shrinks infeasible sources, which assumes "
                "dropping elements can only reduce violation — true for knapsack-style "
                "resource constraints, inverted for covering-style minimization problems "
                "(dropping columns UNcovers rows), where it would silently drive sources "
                "toward the empty set. Use scout_bee_phase_version=1."
            )

    def set_problem(self, problem):
        super().set_problem(problem)
        if problem is not None:
            # optimization_type="auto" only resolves here, so re-check
            self._check_scout_v2_premise()

    # ─────────────────────── distort operators ─────────────────────────

    def distort_operator(self, source, neighbor):
        distort = getattr(self, self.DISTORT_OPERATOR_VERSIONS[self.distort_operator_version])
        return distort(self.food_sources[source], self.food_sources[neighbor])

    def _distort_towards_or_away(self, f1, f2):
        """Version 1: f1 either moves toward f2 or toward its complement,
        keeping whichever candidate Deb's rule prefers.
        """
        r = self.rng.random()
        v_toward = ops.scalar_mul(self.rng, r * self.beta1, f2 - f1)
        v_away = ops.scalar_mul(self.rng, r * self.beta1, (1 - f2) - f1)

        y1 = ops.apply_velocity(f1, v_toward)
        y2 = ops.apply_velocity(f1, v_away)
        return self._better_candidate(
            (y1, *self.evaluate(y1)),
            (y2, *self.evaluate(y2)),
        )

    def _distort_drop_common_add_rare(self, f1, f2):
        """Version 3: move toward f2, optionally drop elements common to both
        (exploration), then try to enrich the winner with rare elements —
        those outside f1 ∪ f2 — via k-tournament.
        """
        common = f1 * f2
        rare = 1 - np.clip(f1 + f2, 0, 1)
        r1, r2, r3 = self.rng.random(3)

        v_toward = ops.scalar_mul(self.rng, r1 * self.beta1, f2 - f1)
        v_drop = ops.remove_elements(self.rng, r2 * self.beta2, common)

        y1 = ops.apply_velocity(f1, v_toward)
        y2 = ops.apply_velocity(f1, ops.add_velocities(v_toward, v_drop))
        best = self._better_candidate(
            (y1, *self.evaluate(y1)),
            (y2, *self.evaluate(y2)),
        )

        y, _, _ = best
        v_add = ops.k_tournament_add(self.rng, r3 * self.beta3, rare, y, self.evaluate, self.k, self.optimization_type)
        y_enriched = ops.apply_velocity(y, v_add)
        return self._better_candidate(best, (y_enriched, *self.evaluate(y_enriched)))

    def _distort_add_rare_on_both(self, f1, f2):
        """Version 4: like version 3, but both branches (with and without the
        drop) are enriched using the same r3 budget for fairness, and Deb's
        rule picks among all four candidates.
        """
        common = f1 * f2
        rare = 1 - np.clip(f1 + f2, 0, 1)
        r1, r2, r3 = self.rng.random(3)

        v_toward = ops.scalar_mul(self.rng, r1 * self.beta1, f2 - f1)
        v_drop = ops.remove_elements(self.rng, r2 * self.beta2, common)

        y1 = ops.apply_velocity(f1, v_toward)
        y2 = ops.apply_velocity(f1, ops.add_velocities(v_toward, v_drop))
        y1e = ops.apply_velocity(
            y1, ops.k_tournament_add(self.rng, r3 * self.beta3, rare, y1, self.evaluate, self.k, self.optimization_type)
        )
        y2e = ops.apply_velocity(
            y2, ops.k_tournament_add(self.rng, r3 * self.beta3, rare, y2, self.evaluate, self.k, self.optimization_type)
        )

        candidates = [(y, *self.evaluate(y)) for y in (y1, y2, y1e, y2e)]
        return self._better_candidate(*candidates)

    def _distort_from_specs(self, f1, f2):
        """Version 5 — the ablation instrument. Each spec in candidate_specs
        names one candidate as the subset of velocity components it applies
        to f1 ("m" move toward f2, "d" drop common elements, "a" add rare
        elements). The default ("m", "md", "ma", "mda") consumes random
        numbers in exactly version 4's order, so it reproduces version 4
        bit-identically; subsets like ("d",) or ("a",) give the
        delete-without-move / add-without-move variants of the ablation, and ("mda",) collapses the operator to a single candidate.

        With shared_add_velocity=True the addition velocity is built once
        per perturbation, ranking entrants by the marginal value on f1, and
        every "a" candidate applies that same velocity — the manuscript's
        formal definition of V3 (and half the tournament evaluations).
        """
        common = f1 * f2
        rare = 1 - np.clip(f1 + f2, 0, 1)
        r1, r2, r3 = self.rng.random(3)

        v_toward = (
            ops.scalar_mul(self.rng, r1 * self.beta1, f2 - f1)
            if any("m" in s for s in self.candidate_specs) else None
        )
        v_drop = (
            ops.remove_elements(self.rng, r2 * self.beta2, common)
            if any("d" in s for s in self.candidate_specs) else None
        )
        v_add_shared = (
            ops.k_tournament_add(self.rng, r3 * self.beta3, rare, f1, self.evaluate, self.k, self.optimization_type)
            if self.shared_add_velocity and any("a" in s for s in self.candidate_specs) else None
        )

        candidates = []
        for spec in self.candidate_specs:
            velocities = [v for flag, v in (("m", v_toward), ("d", v_drop)) if flag in spec]
            y = ops.apply_velocity(f1, ops.add_velocities(*velocities)) if velocities else f1
            if "a" in spec:
                v_add = (
                    v_add_shared
                    if v_add_shared is not None
                    else ops.k_tournament_add(
                        self.rng, r3 * self.beta3, rare, y, self.evaluate, self.k, self.optimization_type
                    )
                )
                y = ops.apply_velocity(y, v_add)
            candidates.append((y, *self.evaluate(y)))
        return self._better_candidate(*candidates)

    def _better_candidate(self, *candidates):
        """Deb's-rule winner among (solution, value, violation) tuples; later
        candidates win ties, so the more exploratory move is preferred.
        """
        best = candidates[0]
        for cand in candidates[1:]:
            if not self.is_first_better(best[1], cand[1], best[2], cand[2]):
                best = cand
        return best

    # ─────────────────────────── scout phase ───────────────────────────

    def _scout_bee_phase(self):
        if self.scout_bee_phase_version == 1:
            super()._scout_bee_phase()
            return

        # Version 2: abandon infeasible sources early by shrinking them
        # (dropping items can only reduce constraint violations); exhausted
        # feasible sources are re-initialized at random as usual.
        # NOTE: the premise holds for knapsack-style resource constraints
        # only. Covering-style problems (e.g. SCP) invert it — dropping
        # elements UNcovers rows — so they must use version 1.
        for n in range(self.num_food_sources):
            if self.food_violations[n] > 0 and self.trials[n] >= 2:
                shrink = min(1.0, self.rng.random() * self.beta2)
                self.food_sources[n, :] = ops.scalar_mul(self.rng, shrink, self.food_sources[n, :])
                self.food_values[n], self.food_violations[n] = self.evaluate_food_source(self.food_sources[n, :])
                self.trials[n] = 0
            elif self.trials[n] > self.limit:
                self._reinitialize_food_source(n)
