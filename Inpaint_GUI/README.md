## How to run Fill Your Face App

To run the Inpainting GUI, cd into this folder in terminal and:    

`./gui_inpaint.py` or `python gui_inpaint.py`

To create an executable, use `python setup.py build`
-- still needs to be fixed as it currently makes a shared library instead of .exe file.
Check the cx-Freeze documentation.

Option to save images after each iteration in GUI, in the gui_inpaint.py file, set self.saveEaItr = True. 
