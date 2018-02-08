PYTHON_ROOT= 

default: install

install:
	${PYTHON_ROOT}python setup.py build build_ext install 

test:
	${PYTHON_ROOT}python runtests.py ${PYTHON_ROOT}nosetests

develop:
	${PYTHON_ROOT}python setup.py build build_ext --inplace
	${PYTHON_ROOT}pip install -e .

doc: 
	cd docs/source && make html

clean:
	${PYTHON_ROOT}python setup.py clean
