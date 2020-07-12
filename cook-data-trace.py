#!/usr/bin/env python3
import csv
from parse import parse
from sys import argv
from os import path
import glob, os
import numpy as np
from pathlib import Path
from itertools import chain

T_SCALE      = 1E-9
NUM_AP       = 15
NUM_JOB_TYPE = 10
RANDOM_SEED  = 11112

SKIP_RAW_PROCESS = True
SKIP_SUMMARY     = False
SKIP_TRACE_GEN   = False

selected_idxs   = range(1)
input_dir       = argv[1] if len(argv)>2 else '../google-datatrace/task_usage/'
output_dir      = './data'
_format         = '{}part-{}-of-{}.csv'
trace_files     = glob.glob( path.join(input_dir,'*.csv') )
trace_files.sort()

#NOTE: Step 1: Raw Processing
if not SKIP_RAW_PROCESS:
    for idx in selected_idxs:
        # generate prefix for current trace file
        trace   = trace_files[idx]
        postfix = path.join( parse(_format, trace)[1] )
        trace_file = path.join(output_dir, 'trace-{}.raw'.format(postfix))
        proc_file  = path.join(output_dir, 'proc-{}.raw'.format(postfix))
        # iteratively reading csv file and summarize
        count = -1
        with open(trace, newline='') as fin, open(trace_file, 'w') as f_trace, open(proc_file, 'w') as f_proc:
            datatrace = csv.reader(fin, delimiter=',')
            ptr = next(datatrace)
            try:
                while(ptr):
                    #collect one sample
                    cur_job = ptr
                    ptr = next(datatrace)
                    while ptr[2]==cur_job[2]: ptr = next(datatrace) #skip tasks in one job
                    start_time= T_SCALE * int(cur_job[0])
                    proc_time = T_SCALE * ( int(cur_job[1]) - int(cur_job[0]) )
                    count = (count+1) % NUM_JOB_TYPE
                    f_trace.write('{:.3f},{}\n'.format(start_time, count)) #(proc_time, job_type)
                    f_proc.write('{:.3f}\n'.format(proc_time)) #(proc_time)
                    pass
            except StopIteration as e:
                print('Part-{:05d} done.'.format(idx))
                pass
            pass
        pass

#NOTE: Step 2: Get statistics
if not SKIP_SUMMARY:
    for idx in selected_idxs:
        # generate trace-*.raw
        trace_in = path.join(output_dir, 'trace-{:05d}.raw'.format(idx))
        trace_record = dict()
        with open(trace_in, 'r') as fin:
            for line in fin.readlines():
                _tmp = line.strip().split(',')
                _item= (_tmp[0], int(_tmp[1]))
                if _item[0] not in trace_record:
                    trace_record[_item[0]] = [0] * NUM_JOB_TYPE
                trace_record[_item[0]][_item[1]] += 1
                pass
            pass
        trace_out= path.join(output_dir, 'trace-{:05d}.stat'.format(idx))
        with open(trace_out, 'w') as fout:
            for i,item in enumerate(sorted(trace_record.items())):
                _data = [min(x,2) for x in item[1]] #normalize
                _data = ','.join(map(str, _data))
                _tmp = ','.join( [str(i), _data] ) #[str(i), _data] / [item[0], _data]
                fout.write(_tmp+'\n')
                pass
            pass

        # generate proc-*.raw
        proc_in = path.join(output_dir, 'proc-{:05d}.raw'.format(idx))
        proc_out= path.join(output_dir, 'proc-{:05d}.stat'.format(idx))
        proc_record = [list() for _ in range(NUM_JOB_TYPE)]
        with open(proc_in, 'r') as fin, open(proc_out, 'w') as fout:
            _data = fin.readlines()
            for i in range(NUM_JOB_TYPE):
                proc_record[i] = sorted([ x.strip() for x in _data[i:][::10] ])
                _val = [float(x) for x in proc_record[i]]
                # _num, _sum = len(_val), sum(_val) #NOTE: (num=11508, avg=0.205)
                # fout.write('%d,%.3f\n'%(_num, _sum/_num))
                fout.write( ','.join(proc_record[i]) + '\n' )
            pass
        pass

#NOTE: Step 3: get arrival probability \lambda_{k,j}, and arrival trace
if not SKIP_TRACE_GEN:
    np.random.seed(RANDOM_SEED)
    for idx in selected_idxs:
        trace_in = path.join(output_dir, 'trace-{:05d}.stat'.format(idx))
        
        trace_folder = path.join(output_dir, 'trace-{:05d}-1x'.format(idx))
        Path( trace_folder ).mkdir(exist_ok=True)
        with open(trace_in, 'r') as fin:
            _statistics = np.zeros((NUM_AP,NUM_JOB_TYPE), dtype=np.int32)
            for line in fin.readlines():
                _idx, _arr_list = line.strip().split(',', 1)
                _arr_list = [int(x) for x in _arr_list.split(',')] #[str --> int]
                _arr_list = [[i]*x for i,x in enumerate(_arr_list)] #[(0,2) --> [0,0]]
                _arr_list = list( chain.from_iterable(_arr_list) )

                _alloc = np.random.permutation(NUM_AP)
                _result = np.zeros((NUM_AP,NUM_JOB_TYPE), dtype=np.int32)
                for i,k in enumerate(_alloc):
                    # print(i, len(_arr_list))
                    if i>len(_arr_list)-1: break
                    _result[k, _arr_list[i]] = 1
                # print(_result)
                _statistics += _result
                _file_name = '{:05d}'.format(int(_idx))
                np.save(Path(trace_folder,_file_name), _result)
                pass
            _statistics = _statistics / np.sum(_statistics)
            _tmp_path   = path.join(trace_folder, 'statistics')
            np.save(_tmp_path, _statistics)
            os.rename(_tmp_path+'.npy', _tmp_path)
            pass

        trace_folder = path.join(output_dir, 'trace-{:05d}-0.25x'.format(idx))
        trace_folder = path.join(output_dir, 'trace-{:05d}-0.5x'.format(idx))
        trace_folder = path.join(output_dir, 'trace-{:05d}-1x'.format(idx))
        trace_folder = path.join(output_dir, 'trace-{:05d}-2x'.format(idx))
        trace_folder = path.join(output_dir, 'trace-{:05d}-3x'.format(idx))
        pass