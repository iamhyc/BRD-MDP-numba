#!/bin/bash
mv -f params-d0.py params.py
make run
mv traces-41122 traces-41122-d0

mv -f params-d1.py params.py
make run
mv traces-41122 traces-41122-d1

mv -f params-d2.py params.py
make run
mv traces-41122 traces-41122-d2

mv -f params-d3.py params.py
make run
mv traces-41122 traces-41122-d3

echo "DONE."