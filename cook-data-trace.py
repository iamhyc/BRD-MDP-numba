#!/usr/bin/env python3
import csv
from sys import argv

file_path = argv[1]
output_dir= argv[2] if argv[2] else './data' 

with open('file_path', newline='') as fd:
    csvreader = csv.reader(fd, delimiter=',', quotechar='|')
    pass