# Assignment 2 - INFOMPRDL
Code for the second practical assignment of Pattern Recognition and Deep Learning (INFOMPRDL)

---
## Installation and Usage

### Clone the repository
```bash
git clone https://github.com/aronnoordhoek/handin-infomprdl-ass2.git
```
### Move to directory
```bash
cd prdl-assignment-2
```
### Open IDE (or find in explorer/IDE)
```bash
code .
```
```bash
pycharm .
```
---
## Data

Download the dataset (to prdl-assignment-2/data)
- [Surfdrive](https://surfdrive.surf.nl/files/index.php/s/3bDWFzLx3smTNTn)

---
The following steps only if your IDE doesn't automatically do this for you:

### Make virtual environment
```bash
python -m venv venv
```
### Activate virtual environment
#### Windows CMD:
```bash
venv\Scripts\activate.bat
```
#### Windows PowerShell:
```bash
venv\Scripts\Activate.ps1
```
#### Unix (Mac, Linux)
```bash
source venv/bin/activate
```

---
## Git

### Get git for Windows should you not have it
[Git for Windows](https://gitforwindows.org/)

### If you don't want to use git CLI / IDE integration
[Github Desktop](https://desktop.github.com/)

- Download, install and open 
- 'Add local repository' (Ctrl+O) 
- Find the 'prdl-assignment-2' you just cloned

---

## Requirements

### Python
Python Version 3.10.11

#### Check python version
```bash
python --version
```
If not 'Python 3.10.11' install from [here](https://www.python.org/downloads/release/python-31011/)
### Dependencies

#### Installing
```bash
pip install -r requirements.txt
```

#### Updating dependencies
```bash
pip freeze > requirements.txt
```
With [Pigar](https://github.com/damnever/pigar) (will overwrite existing reqs)
```bash
pigar generate
```

## Type Checking ([mypy](https://mypy.readthedocs.io/en/stable/command_line.html))
```bash
pip install mypy
```
### All files
```bash
mypy .
```
### Specific script
```bash
mypy path/to/script.py
```

---

## Links

### Data
- [Surfdrive (password: 123)](https://surfdrive.surf.nl/files/index.php/s/3bDWFzLx3smTNTn)

### Report
- [Google Scholar with UU link](https://scholar.google.com/?inst=7240083048524121927)
- [LaTeX Table Generator](https://www.tablesgenerator.com/)


### Setting up GPU
- [Installing CUDA (windows)](https://youtu.be/r7Am-ZGMef8?si=GkoFye4OcPhgbKGK)
- [Generate Tensorflow PIP command for Cuda](https://www.tensorflow.org/install/pip#windows-native)
- [Generate Pytorch PIP command for Cuda](https://pytorch.org/)

---

## Authors
- Panagiotis Aronis (9398333)
- Martino Fabiani (2257742)
- Aron Noordhoek (6733174)
- TODO (#)
- TODO (#)
- TODO (#)
