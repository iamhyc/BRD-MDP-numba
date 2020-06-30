

all:
	pip3 install numpy numba scipy matplotlib PyQt5 termcolor parse --user -i https://mirrors.sustc.us/pypi/simple

run:
	@python3 ./online_main.py --postfix test

submit:
	bsub -q short -n 40 -R "span[ptile=40]" -e %J.err -o %J.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py"

watch:
	watch -n 1 bjobs

clean:
	@rm -f *.err *.out
	@rm -f logs/$(ID).npz logs/$(ID)-test.log
	@rm -rf traces-$(ID)-test