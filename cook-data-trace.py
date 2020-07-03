#!/usr/bin/env python3
import csv
from parse import parse
from sys import argv
from os import path
import argparse

input_dir       = '../google-datatrace/task_usage/'
output_dir      = './data'
_format = '{}-of-{}.csv'
data_file_path  = argv[1]
_prefix         = parse(_format, data_file_path)[0]
cooked_file_path= path.join(output_dir, _prefix, '.txt')

with open(data_file_path, newline='') as fin, open(cooked_file_path, 'w') as fout:
    num = 0
    datatrace = csv.reader(fin, delimiter=',', quotechar='|')
    
    for data in datatrace:
        num += 1
        pass
    pass