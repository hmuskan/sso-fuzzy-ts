#!/bin/bash

python 1.preprocessing.py
python 2.cleaning.py
python 3.feature_extraction.py
python 4.weighted_features.py
python 5.fuzzy_scoring.py
cat 6.summarized/001.txt
