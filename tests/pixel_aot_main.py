"""Entry point for the pixel_aot CTest lines: run tests/pixel_aot_check.py as __main__ through the import system.

Python never caches the bytecode of the script it is started with, so launching the 25k-line
pixel_aot_check.py by path recompiles it on every start. runpy loads it as a module instead, which
reuses a valid .pyc (the Win2025 runner precompiles tests/ into PYTHONPYCACHEPREFIX; -B still reads it).
Arguments, exit status and the exit-time form-coverage record stay those of pixel_aot_check.__main__.
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("pixel_aot_check", run_name="__main__", alter_sys=True)
