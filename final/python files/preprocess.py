
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
import re

import string
#from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import mean_squared_error


# In[101]:


#read json file 
df = pd.read_json('../csv/Health_and_Personal_Care_5.json',lines=True)


# In[102]:


df.head()


# In[103]:


#save initial json file data frame to csv file..
df.to_csv('../csv/Health_and_Personal_Care_5.csv',index=False,sep=',')


# In[4]:


#read csv file 
df=pd.read_csv('../csv/Health_and_Personal_Care_5.csv')


# In[5]:


#all columns of dataframe...
df.columns


# In[6]:


#total users....
print("Total users ",df.reviewerID.unique().size)


# In[7]:


#total products....
print("Total products ",df.asin.unique().size)


# In[8]:


#size of dataframe
print(df.shape)


# In[9]:


type(df)


# In[10]:


def fun1(x):
    x=x.replace('[','')
    x=x.replace(']','')
    x=x.replace("'",'')
    x=x.split(sep=',')

    return int(x[0])

def fun2(x):
    x=x.replace('[','')
    x=x.replace(']','')
    x=x.replace("'",'')
    x=x.split(sep=',')

    return int(x[1])


# In[11]:


#feature extraction from helpful column 
df['helpful_numerator']=df.helpful.apply(lambda x: fun1(x))
df['helpful_denominator']=df.helpful.apply(lambda x: fun2(x))
#drop helpful column
df.drop('helpful',inplace=True,axis=1)


# In[12]:


df.head(n=2)


# In[14]:


# etting description of dataset
df.describe()


# In[15]:


df.head(n=2)


# In[113]:


rating_only=df[['asin','reviewerID','overall']]


# In[114]:


rating_only.to_csv('../csv/rating_only.csv',index=False,sep=',')


# In[16]:


users=df[['reviewerID','reviewerName']]


# In[17]:


x=users.groupby(['reviewerID','reviewerName']).count()


# In[18]:


reviewer=x.reset_index()
reviewer.to_csv('../csv/reviewers.csv',index=False)


# In[19]:


reviewer.head()
reviewer.to_csv('reviewers.csv',index=False)


# In[20]:


df


# In[23]:


count=df.groupby('asin').count()['overall']


# In[25]:


#count


# In[122]:


dfMerged=pd.merge(df,count,how='right',on='asin')


# In[130]:


print(dfMerged.dtypes)


# In[132]:


dfMerged.head()


# In[134]:


dfNew = dfMerged[['asin','summary','overall',"count"]]


# In[136]:


dfNew.head()


# In[138]:


dfNew.to_csv('../csv/product_product.csv',index=False,sep=',')

