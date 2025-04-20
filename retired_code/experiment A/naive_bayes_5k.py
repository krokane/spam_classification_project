#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 09:57:56 2025

@author: kevin
"""

import pandas as pd
import re
import sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import MultinomialNB

train = pd.read_csv('/Users/kevin/Desktop/DSCI599/Projects/Spam Dataset/Combined CSVs/train_data.csv')
test = pd.read_csv('/Users/kevin/Desktop/DSCI599/Projects/Spam Dataset/Combined CSVs/test_data.csv')
train = train.sample(n=5000, random_state=1)

#token and data cleaning function
def clean_text(text):
    #removes body prefix
    if type(text) != str:
        return 'gibberishnonsensenothingeverseenbefore'
    #replaces b's at beginning of body paragraphs
    if text[:2] == "b'" or text[:2] == 'b"':
        text = text[2:]
        text = text[:-1]
    #replaces digit with digit tags
    text = re.sub(r"\d", "<digit>", text)
    #removes new line symbols
    text = re.sub(r"\\n", " ", text)
    #removes periods, commas, dashes, apostrophes, quotes, and new lines
    text = re.sub(r"[.,:'/\-(){}[\]\"]", "", text)
    text = re.sub(r'"', "", text)
    #removes case
    text = text.lower()
    return text.split()
    

#deploy function, add to DFs
train_cleaned = []
for i in range(len(train)):
    train_cleaned.append(clean_text(train.iloc[i]['body']))
test_cleaned = []
for i in range(len(test)):
    test_cleaned.append(clean_text(test.iloc[i]['body']))
train['clean'] = train_cleaned
test['clean'] = test_cleaned


#vectorize data
vectorizer = CountVectorizer(analyzer=lambda x: x)
train_X = vectorizer.fit_transform(train['clean'])
test_X = vectorizer.transform(test['clean'])

tr_binary_labels = []
te_binary_labels = []
for i in range(len(train)):
    if train.iloc[i]['label'] == 'ham':
        tr_binary_labels.append(0)
    else: tr_binary_labels.append(1)
for i in range(len(test)):
    if test.iloc[i]['label'] == 'ham':
        te_binary_labels.append(0)
    else: te_binary_labels.append(1)
train_Y = tr_binary_labels
test_Y = te_binary_labels


#Naive Bayes Model
model = MultinomialNB()

fit = model.fit(train_X, train_Y)

predictions = fit.predict(test_X)

accuracy = accuracy_score(test_Y, predictions)
precision = precision_score(test_Y, predictions)
recall = recall_score(test_Y, predictions)

print(f'Accuracy: {100*round(accuracy,4)}%')
print(f'Precision: {100*round(precision,4)}%')
print(f'Recall: {100*round(recall,4)}%')

