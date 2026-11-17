#!/bin/bash
# One BLAS thread per worker: N workers x 16 threads oversubscribes the 16 cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
exec .venv/bin/python "$@"
