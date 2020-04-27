# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 20:39:30 2020

@author: MLCMOG001
"""
pwd()

#Import the test set
vac_tweets = pd.read_csv("Test.csv")

#This is a df for the results 
y_out=vac_tweets


x_features = vac_tweets['safe_text'].values
 
v_processed_features = []


for sentence in range(0, len(x_features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(x_features[sentence]))

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

    v_processed_features.append(processed_feature)

v_processed_features = vectorizer.transform(v_processed_features).toarray()    


#do this after cleaning
results = text_classifier.predict(v_processed_features)

y_out['label']=results
y_out=y_out.drop(['safe_text'],axis=1)

y_out.to_csv('results.csv',index=False)



