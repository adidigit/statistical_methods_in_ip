import json

def read_ipynb(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        return json.load(f)

first_notebook =  read_ipynb('Q1.ipynb')

second_notebook = read_ipynb('Spring2022_KDE_NormalizingFlows.ipynb')
import copy
final_notebook = copy.deepcopy(first_notebook)
final_notebook['cells'] = first_notebook['cells'] + second_notebook['cells']

def write_ipynb(notebook, notebook_path):
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f)

# Saving the resulting notebook
write_ipynb(final_notebook, 'final_notebook.ipynb')

