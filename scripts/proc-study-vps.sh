#!/bin/bash
# WARNING: Execute this script from parent folder.
rm -f logs/*.npz
mv -f params.py params.py.bak

# submit 10-15 processing-study task
cp -f __test__/params-proc-10-15.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-10-15"
sleep 40

# submit 15-20 processing-study task
cp -f __test__/params-proc-15-20.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-15-20"
sleep 40

# submit 20-25 processing-study task
cp -f __test__/params-proc-20-25.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-20-25"
sleep 40

# submit 25-30 processing-study task
cp -f __test__/params-proc-25-30.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-25-30"
sleep 40

# submit 30-35 processing-study task
cp -f __test__/params-proc-30-35.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-30-35"
sleep 40

# submit 35-40 processing-study task
cp -f __test__/params-proc-35-40.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-35-40"
sleep 40