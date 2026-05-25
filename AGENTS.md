# AGENTS.md

## 1. Project Overview

- A toolkit that incorporates several open-source PYTHON codes in cosmological large-scale structure analysis.
- Python virual environment manager: miniforge. Python version: 3.12
- Using `numpy`, `scipy`, `healpy` as priority.

## Commands
- Dependence installation with conda: `mamba install -c conda-forge`
- Dependence installation with pypi: `pip install`
- Environment activate: `mamba activate`

## Architecutre
- Unified input data structure in `lss_tk/database.py`
- Correlation functions in `lss_tk/*_corr.py`
- Power spectra in `lss_tk/*power.py`
- Detailed structures in `docs/architecture.md`