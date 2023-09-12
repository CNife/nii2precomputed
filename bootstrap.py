import sys
import os
import runpy

cwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cwd)
runpy.run_module("convert_to_precomputed", run_name="__main__")
