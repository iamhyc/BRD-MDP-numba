#!/bin/bash
# WARNING: Execute this script from parent folder.

# submit 1.00x arrival-study task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival.err -o %J-arrival.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-1x\
                        --inject \"TRACE_FOLDER=\"./data/trace-00000-1x\"; arr_prob=np.load(Path(TRACE_FOLDER, \"statistics\"))\""

# submit 1/2 x arrival-study task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival.err -o %J-arrival.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-1_2x\
                        --inject \"TRACE_FOLDER=\"./data/trace-00000-1_2x\"; arr_prob=np.load(Path(TRACE_FOLDER, \"statistics\"))\""

# submit 1/3 x arrival-study task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival.err -o %J-arrival.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-1_3x\
                        --inject \"TRACE_FOLDER=\"./data/trace-00000-1_3x\"; arr_prob=np.load(Path(TRACE_FOLDER, \"statistics\"))\""

# submit 1/4 x arrival-study task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival.err -o %J-arrival.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-1_4x\
                        --inject \"TRACE_FOLDER=\"./data/trace-00000-1_4x\"; arr_prob=np.load(Path(TRACE_FOLDER, \"statistics\"))\""

# submit 1/5 x arrival-study task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-arrival.err -o %J-arrival.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix arrival-1_5x\
                        --inject \"TRACE_FOLDER=\"./data/trace-00000-1_5x\"; arr_prob=np.load(Path(TRACE_FOLDER, \"statistics\"))\""