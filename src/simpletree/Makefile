all: dp.so

dp.so: dp_methods.c dp_methods.h dp_module.c dp_module.h
	rm dp.so 2>/dev/null; python setup.py build_ext --inplace; rm -Rf build/ 2>/dev/null
