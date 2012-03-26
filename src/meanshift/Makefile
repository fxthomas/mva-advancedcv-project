all: edison.so

edison.so: edison_wrapper.cpp edison_wrapper.h
	rm edison.so 2>/dev/null; python setup.py build_ext --inplace; rm -Rf build/ 2>/dev/null
