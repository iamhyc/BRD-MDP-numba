#!/usr/bin/env python3
import os, sys, random

random.seed(1112)
one_shot_list = [random.randint(2**2, 2**16) for _ in range(100)]

template = '''
bsub -q short -n 40 -R "span[ptile=40]" \
-e %J-ser.err -o %J-ser.out \
"NUMBA_NUM_THREADS=40 ./online_main.py --postfix {postfix} --one-shot {one_shot} \
--inject 'TRACE_FOLDER=\"./data/trace-00000-1_2x\"; arr_prob=np.load(Path(TRACE_FOLDER, \"statistics\"))'"
'''.strip()

for num in one_shot_list:
    command = template.format(
        postfix="test",
        one_shot=num
    )
    os.system(command)
    pass

print("[%d] All jobs submitted."%len(one_shot_list))
