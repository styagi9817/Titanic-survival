#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('C:\\Users\\Shubham PC\\Desktop\\titanic.csv')
df


# In[2]:


n_df=df.drop(columns=['Cabin','Survived','Name','SibSp','Parch','Ticket','Embarked'])
n_df


# In[3]:


target=df.drop(columns=['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked','Cabin'])
target


# In[4]:


a_df=n_df.interpolate()
a_df


# In[5]:


from sklearn.preprocessing import LabelEncoder


# In[6]:


le_sex=LabelEncoder()


# In[7]:


a_df['n_sex']=le_sex.fit_transform(a_df['Sex'])
a_df.head()


# In[8]:


b_df=a_df.drop(columns=['Sex'])
b_df


# In[ ]:





# In[9]:


from sklearn import tree


# In[11]:


model = tree.DecisionTreeClassifier()


# In[ ]:





# In[13]:


model.fit(b_df,target)


# In[14]:


model.score(b_df,target)


# In[15]:


model.predict([[887,2,27.0,13.0000,1]])


# In[17]:


from sklearn.linear_model import LogisticRegression


# In[28]:


model=LogisticRegression(C=1)


# In[29]:


model.fit(b_df,target)


# In[30]:


model.score(b_df,target)


# In[25]:


import pickle


# In[27]:


with open('model_titanic','wb') as f:
    pickle.dump(model,f)


# In[ ]:




