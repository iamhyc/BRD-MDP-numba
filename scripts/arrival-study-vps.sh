#!/bin/bash
# WARNING: Execute this script from parent folder.

mv -f params.py params.py.bak

# submit 1-st arrival-study task
cp -f __test__/params-arrival-1st.py params.py && \
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival1.err -o %J-arrival1.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-1st"
sleep 40
