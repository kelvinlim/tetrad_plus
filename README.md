# tetrad_plus

This project provides a python interface to the java Tetrad program from Carnegie Mellon University (https://github.com/cmu-phil/tetrad).

The primary motivation of this project was to provide a set of commands that could be used in a jupyter notebook. Combined with the dgraph_flex package, it supports the interactive use of causal discovery algorithms such as gfci, running of a SEM using the causal graph and data, and creation of publication quality graphs using the graphviz program.

The code has been designed and tested to run on Windows11, macOS Sequoia and Ubuntu 22.04.  It should run on other versions of these platforms.

For a simple sample usage, try out the tetrad_demo.ipynb file in the github repository. This will run within vscode.

You will need a JDK21 or higher version which can be downloaded from here: https://www.oracle.com/java/technologies/downloads/#java21

You will also need the graphviz package which can be downloaded from here: https://graphviz.org/download/

## Python Environment Creation

It is highly recommended you create a virtual environment for running python.

```
# Create a virtual environment (if you don't have one)
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Then install the necessary packages using pip
# On Windows:
pip install -r requirements_win11.txt
# On macOS/Linux:
pip install -r requirements.txt
```
