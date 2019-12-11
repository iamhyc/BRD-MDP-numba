#!/usr/bin/env python3
import json

#------------------------------------------- Step 1 -------------------------------------------#
#NOTE: sort and merge the data
input_fh  = open('./cooked-google-trace.txt', 'r')
raw_data = input_fh.readlines()
raw_data = raw_data[1:]

timestamp = dict()
for line in raw_data:
    _, t_stamp, t_proc = line.split()
    t_stamp, t_proc = float(t_stamp), float(t_proc)
    if t_stamp in timestamp:
        timestamp[t_stamp].append(t_proc)
        timestamp[t_stamp].sort()
    else:
        timestamp[t_stamp] = [t_proc]
    pass

#NOTE: filter the processing time larger than 30 seconds
for key in timestamp.keys():
    timestamp[key] = list( filter(lambda x:x<30.0, timestamp[key]) )
    pass

#NOTE: calculate the arrival rate dict


keys = list(timestamp.keys())
keys.sort()
with open('./sorted-trace.txt', 'w+') as fd:
    fd.write(str(len(keys)) + '\n')
    for key in keys:
        timestamp[key].sort()
        fd.write('{key} {num}\n{list}\n\n'.format(
            key = key,
            num = len(timestamp[key]),
            list= ' '.join([str(x) for x in timestamp[key]])
        ))
    pass

#------------------------------------------- Step 2 -------------------------------------------#
#NOTE: difference the data, scale the number by 100:1, and calculate the frequency scale the data by 100:1
