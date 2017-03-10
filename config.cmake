#config values
option(GENERATE_PYTHON_CALLERS  "generate command line scripts to call python scripts" ON)
set(PYTHON_EXECUTABLE           "" CACHE PATH "vanilla python executable")
set(PYTHON_EXECUTABLE_FOR_SETUP "" CACHE PATH "python executable for use with setup.py (if using WinPython, must be \"WinPython Interpreter.exe\"")
set(CLIPPER_BASE_DIR            "" CACHE PATH "tweaked clipper/iopaths base dir")
