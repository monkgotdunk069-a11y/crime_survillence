"""
Script to execute project.ipynb and save it with complete cell outputs, tables and plots.
"""

import nbformat
from nbclient import NotebookClient


def run_notebook():
    print("[1/2] Reading project.ipynb...")
    nb = nbformat.read('project.ipynb', as_version=4)

    print("[2/2] Executing all cells in project.ipynb...")
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()

    with open('project.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("[OK] project.ipynb fully executed and updated with all visual outputs!")


if __name__ == '__main__':
    run_notebook()
