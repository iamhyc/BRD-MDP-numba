
all:
	pip3 install --user numpy
	pip3 install --user numba
	pip3 install --user scipy
	pip3 install --user matplotlib

run:
	python3 ./online_main.py

clean:
	@rm -rf logs/
	@rm -rf traces-*