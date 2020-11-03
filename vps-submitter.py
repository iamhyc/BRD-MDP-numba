#!/usr/bin/env python3
import os, sys, random
from threading import main_thread

global_template = '''
bsub -q short -n 40 -R "span[ptile=40]" \
-e %J-ser.err -o %J-ser.out \
"NUMBA_NUM_THREADS=40 ./online_main.py --postfix {postfix} --one-shot {one_shot} \
--inject 'TRACE_FOLDER=\"./data/trace-00000-1_2x\"; arr_prob=np.load(Path(TRACE_FOLDER, \"statistics\"))'"
'''.strip()

local_template = '''
./online_main.py --postfix {postfix} --one-shot {one_shot} \
--inject 'TRACE_FOLDER=\"./data/trace-00000-1_2x\"; arr_prob=np.load(Path(TRACE_FOLDER, \"statistics\"))'
'''

if __name__ == "__main__":
    try:
        NUM = int(sys.argv[1])
        random.seed(1112)
        one_shot_list = [random.randint(2**2, 2**16) for _ in range(NUM)]

        for num in one_shot_list:
            command = global_template.format(
                postfix="test",
                one_shot=num
            )
            os.system(command)
        
        print("[%d] All jobs submitted."%len(one_shot_list))
    except Exception as e:
        print(e)
    finally:
        pass
