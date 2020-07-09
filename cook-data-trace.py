#!/usr/bin/env python3
import csv
from parse import parse
from sys import argv
from os import path
import glob

T_SCALE = 1E-9
NUM_JOB_TYPE = 10

selected_idxs   = range(3)
input_dir       = argv[1] if len(argv)>2 else '../google-datatrace/task_usage/'
output_dir      = './data'
_format         = '{}part-{}-of-{}.csv'
trace_files     = glob.glob( path.join(input_dir,'*.csv') )
trace_files.sort()

#NOTE: Step 1: Raw Processing
for idx in selected_idxs:
    # generate prefix for current trace file
    trace   = trace_files[idx]
    postfix = path.join( parse(_format, trace)[1] )
    trace_file = path.join(output_dir, 'trace-{}.raw'.format(postfix))
    proc_file  = path.join(output_dir, 'proc-{}.raw'.format(postfix))
    # iteratively reading csv file and summarize
    count = 0
    with open(trace, newline='') as fin, open(trace_file, 'w') as f_trace, open(proc_file, 'w') as f_proc:
        datatrace = csv.reader(fin, delimiter=',')
        ptr = next(datatrace)
        while(ptr):
            #collect one sample
            cur_job = ptr
            ptr = next(datatrace)
            while ptr[2]==cur_job[2]: ptr = next(datatrace) #skip tasks in one job
            start_time= T_SCALE * int(cur_job[0])
            proc_time = T_SCALE * ( int(cur_job[1]) - int(cur_job[0]) )
            count = (count+1) % NUM_JOB_TYPE
            f_trace.write('{:.3f},{}\n'.format(start_time, count)) #(proc_time, job_type)
            f_proc.write('{:.3f},{}\n'.format(proc_time, count)) #(proc_time, job_type)
            pass
        pass
    pass

#NOTE: Step 2: Get statistics
