all: simpletree.so

simpletree.so: simpletree_funcs.c simpletree_funcs.h simpletree_mod.c simpletree_mod.h
	rm simpletree.so; python setup.py build_ext --inplace
