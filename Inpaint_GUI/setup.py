import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["tkinter","tensorflow", "numpy", "PIL","preinpaint", "inpaint", "postinpaint", "scipy.misc", "skimage.io", "imageio", "functions" ],
"include_files": ["paint.ico", "about.txt"]}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(  name = "InpaintGUI",
        version = "1.2",
        description = "Fill Your Face Application",
        options = {"build_exe": build_exe_options},
        executables = [Executable("gui_inpaint.py", base=base, icon="paint.ico")])
