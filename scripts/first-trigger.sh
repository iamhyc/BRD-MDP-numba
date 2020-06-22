#!/bin/bash
# WARNING: Execute this script from parent folder.
cp -f params.py __test__/params-delay-full.py
cp -f params.py __test__/params-delay-normal.py
cp -f params.py __test__/params-delay-fixed.py

cp -f params.py __test__/params-arrival-1.50.py #1
cp -f params.py __test__/params-arrival-1.70.py #2
cp -f params.py __test__/params-arrival-1.90.py #3
cp -f params.py __test__/params-arrival-2.10.py #4
cp -f params.py __test__/params-arrival-2.30.py #5
cp -f params.py __test__/params-arrival-2.50.py #6
cp -f params.py __test__/params-arrival-2.70.py #7
cp -f params.py __test__/params-arrival-2.90.py #8

cp -f params.py __test__/params-arrival-10-15.py #1
cp -f params.py __test__/params-arrival-15-20.py #2
cp -f params.py __test__/params-arrival-20-25.py #3
cp -f params.py __test__/params-arrival-25-30.py #4
cp -f params.py __test__/params-arrival-30-35.py #5
cp -f params.py __test__/params-arrival-35-40.py #6

echo "Folder __test__ Intialized."