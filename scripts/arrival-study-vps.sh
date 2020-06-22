#!/bin/bash
# WARNING: Execute this script from parent folder.
rm -f logs/*.npz
mv -f params.py params.py.bak

# submit 1.50 arrival-study task
cp -f __test__/params-arrival-1.50.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival.err -o %J-arrival.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-1.50"
sleep 40

# submit 1.70 arrival-study task
cp -f __test__/params-arrival-1.70.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival.err -o %J-arrival.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-1.70"
sleep 40

# submit 1.90 arrival-study task
cp -f __test__/params-arrival-1.90.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival.err -o %J-arrival.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-1.90"
sleep 40

# submit 2.10 arrival-study task
cp -f __test__/params-arrival-2.10.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival.err -o %J-arrival.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-2.10"
sleep 40

# submit 2.30 arrival-study task
cp -f __test__/params-arrival-2.30.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival.err -o %J-arrival.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-2.30"
sleep 40

# submit 2.50 arrival-study task
cp -f __test__/params-arrival-2.50.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival.err -o %J-arrival.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-2.50"
sleep 40

# submit 2.70 arrival-study task
cp -f __test__/params-arrival-2.70.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival.err -o %J-arrival.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-2.70"
sleep 40

# submit 2.90 arrival-study task
cp -f __test__/params-arrival-2.90.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival.err -o %J-arrival.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-2.90"
sleep 40