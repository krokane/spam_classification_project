#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 20:22:38 2025

@author: kevin
"""

import email
import pandas as pd
import re

from email.policy import default
from pathlib import Path

#Hugging Face Dataset
hf_df = pd.read_csv('/Users/kevin/Desktop/DSCI599/Projects/Spam Dataset/Hugging Face/Phishing_Email.csv')

hf_df_label = []
for i in range(len(hf_df)):
    if hf_df.iloc[i]['Email Type'] == 'Safe Email':
        hf_df_label.append('ham')
    else: hf_df_label.append('spam')
hf_df['label'] = hf_df_label
hf_df['body'] = hf_df['Email Text']

hugging_face_data = hf_df[['body','label']]
hugging_face_data.to_csv('hugging_face_data.csv')
#18650 -- 11322 ham, 7328 spam

#Spam Assasin Dataset
SA_df = pd.DataFrame(columns = ['subject','sender', 'body', 'label'])
folder = Path('/Users/kevin/Desktop/DSCI599/Projects/Spam Dataset/SpamAssassin')
for file in folder.rglob('*.*'):
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        msg = email.message_from_file(f, policy=default)
    subject = msg['Subject']
    sender = msg['From']
    body = msg.get_payload(decode=True)
    label = str(file).split('/')[-2]
    SA_df = pd.concat([SA_df, pd.DataFrame([[subject,sender,body,label]], columns=SA_df.columns)], ignore_index=True)
SA_df = SA_df.drop(0)

SA_df_label = []
for i in range(len(SA_df)):
    if re.search('ham',SA_df.iloc[i]['label']): 
        SA_df_label.append('ham')
    else: SA_df_label.append('spam')
SA_df['label'] = SA_df_label

spam_assassin_data = SA_df[['subject','sender','body','label']]
spam_assassin_data.to_csv('spam_assassin_data.csv')

#Has subject, sender, body, and label (spam/ham file names)
#6,097 -- 4201 ham, 1897 spam

#Enron Dataset
enron_df = pd.DataFrame(columns = ['subject','body','label'])
folder2 = Path('/Users/kevin/Desktop/DSCI599/Projects/Spam Dataset/Enron')
for file in folder2.rglob('*.*'):
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        msg = email.message_from_file(f, policy=default)
    subject = msg['Subject']
    body = msg.get_payload(decode=True)
    label = str(file).split('/')[-2]
    enron_df = pd.concat([enron_df, pd.DataFrame([[subject,body,label]], columns=enron_df.columns)], ignore_index=True)
enron_df = enron_df[(enron_df['label'] == 'spam') | (enron_df['label'] == 'ham')]

enron_data = SA_df[['subject','body','label']]
enron_data.to_csv('enron_data.csv')
#Has subject, body, label(spam/ham)
#33,725 -- spam 17171 ham 16545

#Combine Datasets
sample_h = hf_df[['body','label']].sample(n=8500, random_state=1)
sample_s = SA_df[['body','label']].sample(n=6000, random_state=1)
sample_e = enron_df[['body','label']].sample(n=8500, random_state=1)

def balanced_samp(df):
    spam = df[df['label'] == 'spam'].sample(n=500, random_state=1)
    ham = df[df['label'] == 'ham'].sample(n=500, random_state=1)
    return pd.concat([spam, ham]).sample(frac=1, random_state=1)

test_h = balanced_samp(sample_h)
test_s = balanced_samp(sample_s)
test_e = balanced_samp(sample_e)

train_h = sample_h.drop(test_h.index)
train_s = sample_s.drop(test_s.index)
train_e = sample_e.drop(test_e.index)

test_data = pd.concat([test_h,test_s,test_e]).sample(frac=1, random_state=1)
train_data = pd.concat([train_h,train_s,train_e]).sample(frac=1, random_state=1)

test_data.to_csv('test_data.csv')
train_data.to_csv('train_data.csv')
