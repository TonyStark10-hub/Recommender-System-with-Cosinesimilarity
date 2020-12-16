#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:02:58 2020

@author: tony
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text=['London Paris London','Paris Paris London']
cv=CountVectorizer()

count_matrix=cv.fit_transform(text)
#print(count_matrix.toarray())
similarity_scores=cosine_similarity(count_matrix)
print (similarity_scores)