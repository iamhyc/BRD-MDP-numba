#!/bin/bash
# WARNING: Execute this script from parent folder.

# submit 1.00x processing-study task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-1.00x \
                        --inject 'PROC_MIN=int(0.75*N_SLT); PROC_MAX=int(1.00*N_SLT); proc_mean=genProcessingParameter(es2ap_map,True)'"

# submit 1.25x processing-study task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-1.25x \
                        --inject 'PROC_MIN=int(1.00*N_SLT); PROC_MAX=int(1.25*N_SLT); proc_mean=genProcessingParameter(es2ap_map,True)'"

# submit 1.50x processing-study task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-1.50x \
                        --inject 'PROC_MIN=int(1.25*N_SLT); PROC_MAX=int(1.50*N_SLT); proc_mean=genProcessingParameter(es2ap_map,True)'"

# submit 1.75x processing-study task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-1.75x \
                        --inject 'PROC_MIN=int(1.50*N_SLT); PROC_MAX=int(1.75*N_SLT); proc_mean=genProcessingParameter(es2ap_map,True)'"

# submit 2.00x processing-study task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-2.00x \
                        --inject 'PROC_MIN=int(1.75*N_SLT); PROC_MAX=int(2.00*N_SLT); proc_mean=genProcessingParameter(es2ap_map,True)'"

# submit 2.25x processing-study task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-2.25x \
                        --inject 'PROC_MIN=int(2.00*N_SLT); PROC_MAX=int(2.25*N_SLT); proc_mean=genProcessingParameter(es2ap_map,True)'"

# submit 2.50x processing-study task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-2.50x \
                        --inject 'PROC_MIN=int(2.25*N_SLT); PROC_MAX=int(2.50*N_SLT); proc_mean=genProcessingParameter(es2ap_map,True)'"

# submit 2.75x processing-study task
bsub -q short -n 40 -R "span[ptile=40]" -e %J-proc.err -o %J-proc.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix proc-2.75x \
                        --inject 'PROC_MIN=int(2.50*N_SLT); PROC_MAX=int(2.75*N_SLT); proc_mean=genProcessingParameter(es2ap_map,True)'"
