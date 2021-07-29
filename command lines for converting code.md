# Converting a Python Script to Jupyter Notebook (with p2j installed)
`$ p2j file_name.py`
# for our project:
`$ p2j PracticeEDA.py`
# and this will overwrite what we had in .ipynb file

# Converting a .jypnb into .py
`$ jupyter nbconvert --to python 'file_name'`
# for our project:
`$ jupyter nbconvert --to python PracticeEDA.ipynb`
# and this will overwrite what we had in .py file