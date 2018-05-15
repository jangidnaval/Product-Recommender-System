
# coding: utf-8

# In[3]:


get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from scipy.spatial.distance import cosine
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import mean_squared_error


# In[7]:


df=pd.read_csv('../csv/rating_only.csv')


# In[8]:


count = df.groupby("asin", as_index=False).count()[['asin','overall']]
#print(count.head())
count.rename(columns={'overall':'count'},inplace=True)
print(count.head())


# In[9]:


count.head()


# In[10]:


mean = df.groupby("asin", as_index=False).mean()
print(mean.head())


# In[15]:


dfMerged = pd.merge(mean,count, on=['asin'])
print(dfMerged.head())


# In[23]:


#taking top 1% products.....
size=int (dfMerged.index.size/100)
print(size)


# In[18]:


dfMerged.sort_values(by='count',ascending=False,inplace=True)


# In[28]:


most_popular=dfMerged.head(n=size)


# In[32]:


most_popular.head(n=10)


# In[31]:


#dfMerged.sort_values(by='overall',ascending=False)

