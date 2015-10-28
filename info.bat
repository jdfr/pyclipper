rem If you already have a installed python distribution,
rem just compile with:
rem     python setup.py build_ext --inplace
rem If not, a good alternative is WinPython. Setting
rem WinPython's base dir in setpath.bat, mk.bat will
rem compile the bindings. However, WinPython's command
rem window will close automatically, so you are unlikely
rem to be able to read any error messages. If you want
rem to be able to read them, do the following:
rem   -Navigate to the base WinPython directory
rem   -Execute "WinPython Command Prompt.exe"
rem   -Navigate to the folder where this file is located
rem   -Execute this: python setup.py build_ext --inplace