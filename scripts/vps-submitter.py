#!/usr/bin/env python3
import os, sys, random

random.seed(1112)
one_shot_list = [random.randint(2**2, 2**16) for _ in range(100)]

command = '''
    bsub -q short -n 40 -R "span[ptile=40]"
    -e %J-ser.err -o %J-ser.out
    "NUMBA_NUM_THREADS=40 ./online_main.py --serial-optimize --postfix serial"
'''.strip('\n')

os.system(command)