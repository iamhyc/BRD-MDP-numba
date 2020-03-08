#!/bin/bash
MAGIC=55132
# cp -f params.py params-d0.py && cp -f params.py params-d1.py && cp -f params.py params-d2.py && cp -f params.py params-d3.py

mv -f params-d0.py params.py && \
make run && \
mv traces-${MAGIC}/summary.npz traces-${MAGIC}/summary && \
mv traces-${MAGIC} traces-${MAGIC}-d0

mv -f params-d1.py params.py && \
make run && \
mv traces-${MAGIC}/summary.npz traces-${MAGIC}/summary && \
mv traces-${MAGIC} traces-${MAGIC}-d1

mv -f params-d2.py params.py && \
make run && \
mv traces-${MAGIC}/summary.npz traces-${MAGIC}/summary && \
mv traces-${MAGIC} traces-${MAGIC}-d2

mv -f params-d3.py params.py && \
make run && \
mv traces-${MAGIC}/summary.npz traces-${MAGIC}/summary && \
mv traces-${MAGIC} traces-${MAGIC}-d3

echo "DONE."