import numpy as np
from pathlib import Path
import joblib  # make sure you have `joblib` installed: pip install joblib


def _read_next_nonempty_line(f):
    # … your existing helper …
    while True:
        pos = f.tell()
        line = f.readline()
        if not line:
            return None
        line = line.strip()
        if line and not line.startswith("+") and not line.lower().startswith("problem"):
            return line


def _read_numbers(f, count):
    # … your existing helper …
    nums = []
    while len(nums) < count:
        line = _read_next_nonempty_line(f)
        if line is None:
            raise EOFError(f"Expected {count} numbers but hit EOF")
        parts = line.split()
        nums.extend(map(float, parts))
    return np.array(nums[:count])


def parse_mknap2_wrapped(file_path, problem_set_name):
    instances = []
    problem_id = 0
    with open(file_path, "r") as f:
        while True:
            hdr = _read_next_nonempty_line(f)
            if hdr is None:
                break
            parts = hdr.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                n, m = map(int, parts)
                profits = _read_numbers(f, m)
                capacity = _read_numbers(f, n)
                weights = np.vstack([_read_numbers(f, m) for _ in range(n)])
                opt_line = _read_next_nonempty_line(f)
                opt = float(opt_line.split()[0])
                problem_id += 1
                instances.append(
                    {
                        "num_of_constraints": n,  # number of constraints
                        "num_of_items": m,  # number of items
                        "profits": profits,
                        "capacity": capacity,
                        "weights": weights,
                        "known_optimum": opt,
                        "problem_name": f"{problem_set_name}-{problem_id}",
                    }
                )
    return instances


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    problem_set_name = "mknap2"
    data_txt = BASE / (problem_set_name + ".txt")
    out_job = BASE / (problem_set_name + ".joblib")

    # 1) Parse
    problems = parse_mknap2_wrapped(data_txt, problem_set_name)
    print(f"Loaded {len(problems)} instances.")

    # 2) Dump to disk
    joblib.dump(problems, out_job)
    print(f"Saved parsed data to {out_job}")

    # 3) (Optional) reload to test
    loaded = joblib.load(out_job)
    print(
        "Reload check:",
        loaded[0]["num_of_items"],
        "items,",
        loaded[0]["num_of_constraints"],
        "constraints, known_optimum=",
        loaded[0]["known_optimum"],
    )
