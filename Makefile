
all:
	pip3 install --user numpy
	pip3 install --user numba
	pip3 install --user scipy
	pip3 install --user matplotlib
	pip3 install --user termcolor

run:
	@python3 ./online_main.py

submit:
	bsub -q short -n 160 -R "span[ptile=40]" -e %J.err -o %J.out "NUMBA_NUM_THREADS=160 python3 ./online_main.py"

clean:
	@rm -f *.err *.out
	@rm -rf logs/$(ID).npz
	@rm -rf traces-$(ID)
