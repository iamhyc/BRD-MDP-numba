#!/bin/bash
# WARNING: Execute this script from parent folder.

# submit large delay support task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-large.err -o %J-large.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix delay-large \
                        --inject 'BR_MIN=int(1.00*N_SLT-1); BR_MAX=int(1.00*N_SLT); br_dist=genDelayDistribution(True)'"
# submit normal-range delay support task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-normal.err -o %J-normal.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix delay-normal \
                        --inject 'BR_MIN=int(0.70*N_SLT-1); BR_MAX=int(0.70*N_SLT); br_dist=genDelayDistribution(True)'"
# submit medium delay task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-medium.err -o %J-medium.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix delay-medium \
                        --inject 'BR_MIN=int(0.50*N_SLT-1); BR_MAX=int(0.50*N_SLT); br_dist=genDelayDistribution(True)'"
                        # submit medium delay task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-small.err -o %J-small.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix delay-small \
                        --inject 'BR_MIN=int(0.10*N_SLT-1); BR_MAX=int(0.10*N_SLT); br_dist=genDelayDistribution(True)'"
