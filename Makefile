
all:
	pip3 install numpy numba scipy matplotlib PyQt5 termcolor --user -i https://mirrors.sustc.us/pypi/simple
	pip3 install networkx --user
	pip3 install parse --user
	pip3 install tqdm --user

run:
	@python3 ./online_main.py --postfix long-time --plot

submit:
	bsub -q short -n 160 -R "span[ptile=40]" -e %J.err -o %J.out "./task-manager.py 60 0"
	#bsub -q short -n 40 -R "span[ptile=40]" -e %J.err -o %J.out "NUMBA_NUM_THREADS=40 python3 ./online_main.py --postfix main"

watch:
	watch -n 1 bjobs

clean:	
	@rm -f *.err *.out
	@rm -f logs/*.err logs/*.out
	@rm -f logs/$(ID).npz logs/$(ID)-test.log
	@rm -rf records-$(ID)