# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:55:35 2020

@author: MLCMOG001
"""
import os
import numpy as np 
import pandas as pd 
import re
import nltk 
nltk.download('stopwords')
  
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#%matplotlib inline

pd.set_option('display.max_columns', None)

#data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
os.chdir ('C:\\Users\\MLCMOG001\\Desktop\\MachineL\\S.Analysis_COVID\\data')
vaccine_tweets = pd.read_csv("Train.csv")

vaccine_tweets.head()
vaccine_tweets.info()

#DROP ROWS 
vaccine_tweets=vaccine_tweets.dropna()
vaccine_tweets.info()

#DROP columns
vaccine_tweets=vaccine_tweets.drop(['tweet_id', 'agreement'], axis=1)

'''
airline_sentiment = vaccine_tweets.groupby(['label']).label.count().unstack()
airline_sentiment.plot(kind='bar')
'''



features = vaccine_tweets['safe_text'].values
labels = vaccine_tweets['label'].values

processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)
    
    
vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()  

len(processed_features[0])
    
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

#TRAIN THE MODEL
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

#MAKE PREDICTIONS
predictions = text_classifier.predict(X_test)

#check performance

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))






