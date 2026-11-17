"""Example experiment: plain binary ABC vs our SBABC on the mknap1 benchmarks.

Run from the repository root:  python main.py
"""

from algorithms import ArtificialBeeColony, SetBasedABC
from core import Config
from experiments import compare_algorithms
from problems.mkp import load_mkp_problems


def main():
    problems = load_mkp_problems("mknap1")

    abc = ArtificialBeeColony(num_food_sources=25, limit=100)

    # best hand-tuned SBABC configuration from the earlier experiments
    sbabc = SetBasedABC(
        topology="von_neumann",
        neighborhood_radius=2,
        num_food_sources=25,
        limit=120,
        beta1=0.5,
        beta2=1.5,
        beta3=3,
        k=5,
        distort_operator_version=4,
    )
    sbabc.name = "SBABC(VN)_v4"

    df = compare_algorithms(
        algorithms=[abc, sbabc],
        problems=problems,
        config=Config(num_runs=5, max_iteration=1000, early_stopping_patience=300, seed=4),
        save_path="results/abc_vs_sbabc_mknap1.csv",
    )

    columns = ["problem", "algorithm", "best_value", "mean_error_pct", "success_runs", "num_runs", "mean_fn_calls"]
    print()
    print(df[columns].to_string(index=False))


if __name__ == "__main__":
    main()
