#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:49:14 2020

@author: tony
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

'''Helper functions '''

#Read CSV file

df=pd.read_csv('movie_dataset.csv')
#print (df.columns)

features=['keywords','cast','genres','director']