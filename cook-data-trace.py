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

SKIP_RAW_PROCESS = False
SKIP_SUMMARY     = False
SKIP_TRACE_GEN   = False
NORM_FACTOR = 250

selected_idxs   = range(1)
input_dir       = argv[1] if len(argv)>2 else '../google-datatrace/task_usage/'
output_dir      = './data'; Path(output_dir).mkdir(exist_ok=True)
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
                print('Part-{:05d} Step 1 done.'.format(idx))
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
                _data = [min(x,NORM_FACTOR) for x in item[1]] #FIXME: normalize
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
        print('Part-{:05d} Step 2 done.'.format(idx))
        pass

#NOTE: Step 3: get arrival trace, and arrival statistics \lambda_{k,j}
Scales = ['1_1x', '1_2x', '1_3x', '1_4x', '1_5x']
zero_padding = np.zeros((NUM_AP,NUM_JOB_TYPE), dtype=np.int32)
if not SKIP_TRACE_GEN:
    np.random.seed(RANDOM_SEED)
    for idx in selected_idxs:
        trace_in = path.join(output_dir, 'trace-{:05d}.stat'.format(idx))
        #NOTE: a) ensure trace folders for all scales
        trace_folders = [path.join(output_dir, 'trace-{:05d}-{}'.format(idx, scale)) for scale in Scales]
        for folder in trace_folders:
            Path( folder ).mkdir(exist_ok=True)
        #NOTE: b) generate statistics for all scales
        _statistics = np.zeros((NUM_AP,NUM_JOB_TYPE), dtype=np.int32)
        with open(trace_in, 'r') as fin:
            for cnt,line in enumerate(fin.readlines()):
                #NOTE: b1) parse from raw trace
                _, _arr_list = line.strip().split(',', 1)
                _arr_list = [int(x) for x in _arr_list.split(',')] #[str --> int]
                _arr_list = [[i]*x for i,x in enumerate(_arr_list)] #[(0,2) --> [0,0]]
                _arr_list = list( chain.from_iterable(_arr_list) )
                #NOTE: b2) allocate arrivals onto APs
                _alloc = np.random.permutation(NUM_AP)
                _result = np.zeros((NUM_AP,NUM_JOB_TYPE), dtype=np.int32)
                for i,k in enumerate(_alloc):
                    if i>len(_arr_list)-1: break
                    _result[k, _arr_list[i]] = 1
                    pass
                _statistics += _result
                #NOTE: b3) save trace*.npy for all scales
                for i,folder in enumerate(trace_folders):
                    _start_idx = cnt*(i+1)
                    _end_idx   = _start_idx + i
                    np.save(Path(folder,'%05d'%_start_idx), _result)
                    for _ in range(_start_idx+1,_end_idx+1):
                        np.save(Path(folder,'%05d'%_), zero_padding)
                        pass
                    pass
                pass
            pass
        #NOTE: c) save statics for all scales
        for scale,folder in zip(Scales,trace_folders):
            _num = len( glob.glob( path.join(folder,'*.npy')) )
            _tmp = _statistics / _num
            _tmp_path   = path.join(folder, 'statistics')
            np.save(_tmp_path, _tmp)
            os.rename(_tmp_path+'.npy', _tmp_path)
            print(_tmp, _num, '\n')
            pass
        print('Part-{:05d} Step 3 done.'.format(idx))
        pass