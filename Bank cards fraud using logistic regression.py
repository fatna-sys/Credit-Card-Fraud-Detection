#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection Model using Logistic Regression

# In[1]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


credit_card_data= pd.read_csv("Desktop/Data projects/Credit Card Fraud Detection/creditcard.csv")


# In[4]:


credit_card_data.head()


# In[5]:


credit_card_data.info()


# In[7]:


credit_card_data.isna().sum()


# In[16]:


credit_card_data["Class"].value_counts()


# In[21]:


legit=credit_card_data[credit_card_data["Class"]==0]
fraud=credit_card_data[credit_card_data["Class"]==1]


# In[27]:


legit.Amount.describe()


# In[28]:


fraud.Amount.describe()


# In[29]:


credit_card_data.groupby('Class').mean()


# In[30]:


legit_sample=legit.sample(n=492)


# In[35]:


new_dataset = pd.concat([legit_sample,fraud], axis = 0)


# In[33]:


new_dataset.Class.value_counts()


# In[36]:


new_dataset.groupby('Class').mean()


# In[37]:


X = new_dataset.drop(columns='Class',axis=1)     # Features
Y = new_dataset['Class']                        #target


# In[39]:


X_train,X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y, random_state=2)


# In[42]:


model=LogisticRegression()


# In[43]:


model.fit(X_train,Y_train)


# In[45]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)


# In[46]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[47]:


print('Accuracy on Training Data: ',training_data_accuracy)
print('Accuracy on Test Data: ',test_data_accuracy)


# In[48]:


model

