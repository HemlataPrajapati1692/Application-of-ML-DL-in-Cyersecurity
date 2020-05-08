#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


dataset=pd.read_csv('/Users/hemi/Downloads/URLdataset.csv')


# In[3]:


dataset.head()


# In[3]:


# Labels
y = dataset.iloc[:,-1]

# Features
url_list = dataset.iloc[:,0]


# In[5]:


# Using Tokenizer
vectorizer = TfidfVectorizer()

# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)


# In[6]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[12]:


sc= StandardScaler(with_mean=False)
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[15]:


logit = LogisticRegression(max_iter=1000)
logit.fit(X_train, y_train)


# In[16]:


#Accuracy of model
print("Accuracy of our model:", logit.score(X_test, y_test))


# In[ ]:




