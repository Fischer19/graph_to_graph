#!/usr/bin/env bash
conda env create -f environment.yml
source activate fischer
pip install -r requirements.txt
python supervised_train.py
