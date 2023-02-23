#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


HouseDF = pd.read_csv('USA_Housing_DATASET_FOR_AI_PROJECT.csv')


# In[5]:


HouseDF.head()


# In[6]:


HouseDF.info()


# In[7]:


HouseDF.describe()


# In[8]:


HouseDF.columns


# In[9]:


sns.pairplot(HouseDF)


# In[10]:


sns.heatmap(HouseDF.corr(), annot=True)


# In[15]:


X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y = HouseDF['Price']


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.40, random_state=101)


# In[20]:


from sklearn.linear_model import LinearRegression 


# In[23]:


lm = LinearRegression() 


# In[24]:


lm.fit(X_train,y_train) 


# In[26]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])


# In[27]:


coeff_df


# In[28]:


predictions = lm.predict(X_test)


# In[29]:


plt.scatter(y_test, predictions)


# In[30]:


sns.distplot((y_test-predictions),bins=50);


# In[31]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))


# In[ ]:




