#!/bin/bash
bsub -q short -n 40 -R "span[ptile=40]" -e %J-ser.err -o %J-ser.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --serial-optimize --postfix serial"
sleep 20
bsub -q short -n 40 -R "span[ptile=40]" -e %J-para.err -o %J-para.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix parallel"