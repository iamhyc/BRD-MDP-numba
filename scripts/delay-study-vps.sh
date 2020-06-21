#!/bin/bash
# WARNING: Execute this script from parent folder.

mv -f params.py params.py.bak

# submit full-range delay support task
cp -f __test__/params-delay-full.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-full.err -o %J-full.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix delay-full"
sleep 40

# submit normal-range delay support task
cp -f __test__/params-delay-normal.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-normal.err -o %J-normal.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix delay-normal"
sleep 40

# submit fixed delay task
cp -f __test__/params-delay-fixed.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-fixed.err -o %J-fixed.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix delay-fixed"
