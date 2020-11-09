#!/usr/bin/env python3
import os, sys, random, time
import subprocess as sp
from threading import main_thread

NUM_SLOT = 10
RANDOM_SEED = 11112

global_template = '''
bsub -q short -n 40 -R "span[ptile=40]" \
-e %J-ser.err -o %J-ser.out \
"NUMBA_NUM_THREADS=40 ./online_main.py --postfix {postfix} --one-shot {one_shot}"
'''.strip()

local_template = [
    "./online_main.py",
    "--postfix", "{postfix}",
    "--one-shot", "{one_shot}",
    "--inject", "TRACE_FOLDER='./data/trace-00000-1_3x'; arr_prob=np.load(Path(TRACE_FOLDER, 'statistics'))"
]

def all_done(cpu_stat):
    for stat in cpu_stat:
        if stat is not None:
            return False
    return True

if __name__ == "__main__":
    try:
        NUM = int(sys.argv[1])
        VERBOSE = False if len(sys.argv)>2 else True
        random.seed(RANDOM_SEED)
        one_shot_list = [random.randint(2**2, 2**16) for _ in range(NUM)]
        start_time = time.time()

        task_list = dict()
        for idx, num in enumerate(one_shot_list):
            command = local_template.copy()
            command[ local_template.index("{postfix}") ]  = "test"
            command[ local_template.index("{one_shot}") ] = '%05d'%(num)
            task_list[idx] = command
            pass
        
        cpu_stat = [None] * NUM_SLOT; cpu_stat[0] = 0
        while not all_done(cpu_stat):
            _empty = 0
            for idx, stat in enumerate(cpu_stat):
                try:
                    if (stat is None) or (stat==0):
                        _key, _command = task_list.popitem()
                        _stdout = open('./logs/%05d-%02d.out'%(RANDOM_SEED, _key), 'w')
                        _stderr = open('./logs/%05d-%02d.err'%(RANDOM_SEED, _key), 'w')
                        cpu_stat[idx] = sp.Popen(_command, stdout=_stdout, stderr=_stderr)
                    else:
                        _ret = stat.poll()
                        if _ret is not None:
                            _time = time.time() - start_time
                            print( "[%.2fs elapsed] one job completed with retcode %r%s"%(_time, _ret, ' '*20) )
                            cpu_stat[idx] = None
                        else:
                            continue
                except KeyError as e:
                    _empty += 1
                pass

            _time = time.time() - start_time
            if VERBOSE:
                print("[%.2fs elapsed] %d job(s) running, %d job(s) pending%s\r"%( _time, NUM_SLOT-_empty, len(task_list), ' '*10), end='')
            time.sleep(1)
            pass
        
        print( "All jobs completed: %.2f second.%s"%(time.time()-start_time, ' '*20) )
        # print("[%d] All jobs submitted."%len(one_shot_list))
    except Exception as e:
        raise e#print(e)
    finally:
        pass
