
import numpy as np
from pathlib import Path
import joblib   # make sure you have `joblib` installed: pip install joblib




def _read_next_nonempty_line(f):
    """Read the next non-blank line from file f."""
    line = f.readline()
    while line and line.strip() == "":
        line = f.readline()
    return line


def _read_numbers(f, count):
    """
    Read numbers (floats) from file f across as many lines as needed
    until we collect exactly `count` values.
    """
    nums = []
    while len(nums) < count:
        line = _read_next_nonempty_line(f)
        if not line:
            raise EOFError(f"Expected {count} numbers but reached EOF")
        parts = line.strip().split()
        nums.extend(map(float, parts))
    if len(nums) > count:
        # If the line provided extra tokens, push them back by repositioning file pointer
        # (optional: depends on whether extra tokens ever occur)
        pass
    return np.array(nums[:count])


def parse_mknap1_wrapped(file_path,problem_set_name):
    """
    Parse the OR-Library mknap1 format, handling lines
    that wrap items/weights across multiple lines.
    Returns a list of dicts with keys: 'n','m','opt','profits','weights','capacity'.
    """
    instances = []
    problem_id=0
    with open(file_path, 'r') as f:
        # Number of instances
        first = _read_next_nonempty_line(f)
        K = int(first.strip().split()[0])

        for _ in range(K):
            # Header line: n, m, optimal value
            header = _read_next_nonempty_line(f).strip().split()
            n, m, opt = int(header[0]), int(header[1]), float(header[2])

            # Read profits (n values)
            profits = _read_numbers(f, n)

            # Read m rows of weights (each with n values)
            weights = np.vstack([_read_numbers(f, n) for __ in range(m)])

            # Read capacities (m values)
            capacity = _read_numbers(f, m)
            problem_id+=1

            instances.append({
                'num_of_items': n,
                'num_of_constraints': m,
                'known_optimum': opt,
                'profits': profits,
                'weights': weights,
                'capacity': capacity,
                'problem_name':f"{problem_set_name}-{problem_id}"
            })
    return instances


# Example usage:
if __name__ == "__main__":
    BASE     = Path(__file__).resolve().parent
    problem_set_name='mknap1'
    data_txt = BASE / (problem_set_name+".txt")
    out_job  = BASE / (problem_set_name+".joblib")


    problems = parse_mknap1_wrapped(data_txt,problem_set_name)
    # Inspect the 4th instance (20 items, 10 constraints)
    p4 = problems[3]
    print("Instance 4:", p4['num_of_items'], "items;",
          p4['num_of_constraints'], "constraints; opt =", p4['known_optimum'])
    print("Profits:", p4['profits'])
    print("Weights row 1:", p4['weights'][0])
    print("Capacities:", p4['capacity'])


      # 2) Dump to disk
    joblib.dump(problems, out_job)
    print(f"Saved parsed data to {out_job}")
