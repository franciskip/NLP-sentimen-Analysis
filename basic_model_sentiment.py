#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Import Libraries

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt    

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")


# In[8]:


# Import data

use_cols = ['airline_sentiment', 'text', 'airline']

df = pd.read_csv('Tweets (1).csv', usecols = use_cols)

df.head()


# In[9]:


# Lets process the text

stop_words = stopwords.words('english')

df['text_without_stopwords'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

countVect = CountVectorizer(min_df= 10)

binaryVector = countVect.fit_transform(df.text_without_stopwords)


# In[10]:


pickle.dump(countVect, open('tranform.pkl', 'wb'))


# In[11]:


# Seperate dataset into test and train

y = df.airline_sentiment
X = binaryVector

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=123)

print([x.shape for x in [train_X, test_X, train_y, test_y]])


# In[20]:


train_y


# In[12]:


# Now for testing the naive bayes model

MNB = MultinomialNB()
MNB.fit(train_X, train_y)

predicted = MNB.predict(test_X)
accuracy_score = metrics.accuracy_score(predicted, test_y)
confusion_count = metrics.confusion_matrix(predicted, test_y)


print('Accuracy: ',accuracy_score,'\n')
print('Confusion Matrix:\n',confusion_count)


# In[18]:


filename = 'nlp_model.pkl'
pickle.dump(MNB, open(filename, 'wb'))


# In[ ]:




