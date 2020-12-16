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
def title_to_index(title):
    return df[df.title==title]['index'].values[0]
def index_to_title(index):
    return df[df.index==index]['title'].values[0]

#Read CSV file

df=pd.read_csv('movie_dataset.csv')
#print (df.columns)

features=['keywords','cast','genres','director']
#replacing the value NAN with ''
for feature in features :
    df[feature]=df[feature].fillna('')
                
#combining function    
def combine_features(row):
    return row['keywords']+' '+row['cast']+' '+row['genres']+' '+row['director']

#creating a new column in dataframe combining all the features
df['combined_features']=df.apply(combine_features,axis=1)

#print(df['combined_features'].head())


#generating a count matrix based one new combined_features column in df
cv=CountVectorizer()
count_matrix=cv.fit_transform(df['combined_features'])
#print(count_matrix)

#generate coine similarity matrix from count matrix
cosine_sim=cosine_similarity(count_matrix)
#take input from user 
fav_movie=input('Enter your favourite movie :')
liked_movie=fav_movie

#get the index of provided movie 
movie_index=title_to_index(liked_movie)

#get similar movie indexes from cosine similarity matrix and convert it into list with enumaration
similar_movies=list(enumerate(cosine_sim[movie_index]))

#Sort the similar movies based on the cosine similarity score in descending order

sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)

#to get 10 top similar movies loop 10 times in sorted list and get the index and convert index into movie title

i=0
for movie in sorted_similar_movies:
    if i>10:
        break
    print(index_to_title(movie[0]))
    i+=1
    




