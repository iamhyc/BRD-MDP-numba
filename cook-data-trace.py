#!/usr/bin/env python3
import csv
from parse import parse
from sys import argv
from os import path
import glob

T_SCALE = 1E-9
NUM_JOB_TYPE = 10
NUM_AP       = 15

SKIP_RAW_PROCESS = True
SKIP_SUMMARY     = False

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
        # process trace-*.raw
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
                _tmp = ','.join( [item[0], _data] ) #[str(idx), _data]
                fout.write(_tmp+'\n')
                pass
            pass
        # process proc-*.raw
        proc_in = path.join(output_dir, 'proc-{:05d}.raw'.format(idx))
        proc_record = [list() for _ in range(NUM_JOB_TYPE)]
        with open(proc_in, 'r') as fin:
            _data = fin.readlines()
            for i in range(NUM_JOB_TYPE):
                proc_record[i] = _data[i:][::10]
            pass
        proc_out= path.join(output_dir, 'proc-{:05d}.stat'.format(idx))
        with open(proc_out, 'w') as fout:
            for i in range(NUM_JOB_TYPE):
                fout.write( str(proc_record[i])+'\n' )
            pass
        pass
