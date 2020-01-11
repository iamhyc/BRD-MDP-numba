#!/bin/bash
# cp -f params.py params-d0.py && cp -f params.py params-d1.py && cp -f params.py params-d2.py && cp -f params.py params-d3.py

mv -f params-d0.py params.py && \
make run && \
mv traces-41122/summary.npz traces-41122/summary && \
mv traces-41122 traces-41122-d0

mv -f params-d1.py params.py && \
make run && \
mv traces-41122/summary.npz traces-41122/summary && \
mv traces-41122 traces-41122-d1

mv -f params-d2.py params.py && \
make run && \
mv traces-41122/summary.npz traces-41122/summary && \
mv traces-41122 traces-41122-d2

mv -f params-d3.py params.py && \
make run && \
mv traces-41122/summary.npz traces-41122/summary && \
mv traces-41122 traces-41122-d3

echo "DONE."