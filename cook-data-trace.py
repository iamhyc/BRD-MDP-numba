#!/usr/bin/env python3
import csv
from parse import parse
from sys import argv
from os import path
import glob

selected_idxs   = range(3)
input_dir       = argv[1] if len(argv)>2 else '../google-datatrace/task_usage/'
output_dir      = './data'
_format         = '{}-of-{}.csv'
trace_files     = glob.glob( path.join(input_dir,'*.csv') )
trace_files.sort()

for idx in selected_idxs:
    # generate prefix for current trace file
    trace   = trace_files[idx]
    _prefix = path.join( parse(_format, trace)[0] )
    _prefix = path.join(output_dir, _prefix)
    trace_file = _prefix+'-trace.txt'; proc_file = _prefix+'-proc.txt'
    # iteratively reading csv file and summarize
    with open(trace, newline='') as fin, open(trace_file, 'w') as f_trace, open(proc_file, 'w') as f_proc:
        datatrace = csv.reader(fin, delimiter=',')
        for data in datatrace:
            #TODO: iteratively processing
            pass
        pass
    pass

# with open(data_file_path, newline='') as fin, open(cooked_file_path, 'w') as fout:
#     num = 0
#     datatrace = csv.reader(fin, delimiter=',')
#     pass