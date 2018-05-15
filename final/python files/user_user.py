
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame 
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


# In[2]:


df=pd.read_csv('../csv/rating_only.csv');


# In[3]:


df.head()


# In[4]:


df.describe()


# In[10]:


df.sort_values('overall').head()


# In[12]:


df.groupby('overall').count()


# In[15]:


print("total size ",df.index.size)


# In[16]:


df.groupby('overall').count()


# In[19]:


#total users....
total_u=df['reviewerID'].unique().size
print(total_u)


# In[17]:


#total products ....
total_p=df['asin'].unique().size
print(total_p)


# In[20]:


#avg total rating per user 
df['overall'].sum()/total_u


# In[21]:


#avg total rating per item 
df['overall'].sum()/total_p


# In[23]:


#product rated by per user
#df.groupby('reviewerID')['overall'].count().sort_values()


# #  euclidean similarity

# In[5]:


df[df['reviewerID']=='ALC5GH8CAMAI7']


# In[6]:


def euc_sim(df,u1,u2):
    products_u1=df[df['reviewerID']==u1]
    products_u2=df[df['reviewerID']==u2]
    
    comman=pd.merge(products_u1,products_u2,how='inner',on='asin')
    print(comman)
    
    if comman.index.size==0:
        return 0
    
    total=((comman.overall_x-comman.overall_y)**2).values.sum()
    #print("Euclidean similarity......",1/(1+total))
    
    # sim-->1 is perfect similar....     sim-->0 not similar......
    return 1/(1+total)
  

def pearson_corr(df,u1,u2):
    products_u1=df[df['reviewerID']==u1]
    products_u2=df[df['reviewerID']==u2]
    
    comman=pd.merge(products_u1,products_u2,how='inner',on='asin')
    #print(comman)

    if comman.index.size==0:
        return 0
    
    avg1=products_u1.overall.sum()/products_u1.index.size
    avg2=products_u2.overall.sum()/products_u2.index.size
    
    t1=comman['overall_x']-avg1
    t2=comman['overall_y']-avg2
    
    #print(t1)
    #print(t2)
    #print(t1.multiply(t2))
    
    s1=sum(t1.multiply(t2))
    #print(s1)
    if s1==0:
        return 0
    
    
    #pearson relation range b/w -1 to 1 . 
    #-1 indicate perfectly negative linear relationship...
    # 0 indicates no linear relationship ....
    #+1 indicate perfectly positive linear relationship...
    
    cor=s1/(((sum(t1**2))**.5) * ((sum(t2**2))**.5))
    #print("pearson_correlation is  ",cor)
    return cor
    
    
    # corr by inbuild formula.....
    #x=comman['rating_x'].corr(comman['rating_y'],method='pearson')
    #print(x)
    #return comman['rating_x'].corr(comman['rating_y'],method='pearson')


# In[77]:


x=euc_sim(df,'A32T585IZ0DJX2','ACKL6OEZEOL45')


# In[78]:


x=pearson_corr(df,'A32T585IZ0DJX2','ACKL6OEZEOL45')


# In[7]:


def my_average(df,user): 
    temp = df[df['reviewerID']==user].overall
    avg = temp.sum()/temp.count()
    return avg


# In[37]:


#calculating  rating od a product  based on similarity with other users....
def calculate_rating(df,user,product):
    # calculate the predicted rating for a product
    numerator = 0
    denominator = 0
    cnt = 0
    for user2 in df['reviewerID'].drop_duplicates():
        rate_v_i_series = df.loc[(df['reviewerID']==user2) & (df['asin']==product)].overall # rating given by user2 to product
        if(rate_v_i_series.empty): # user has not rated the product
            continue
        cnt += 1
        rate_v_i = rate_v_i_series.iloc[0]
        
        avg_v = my_average(df,user2) # average rating given by user2
        
        weight = pearson_corr(df,user, user2) # Pearson correlation between user and user2
        
        numerator += (rate_v_i - avg_v)*weight
        
        denominator += weight
    
    rating_i = numerator/denominator + my_average(df,user) # Predicted rating of movie i 
    return rating_i


# In[9]:


from sklearn.model_selection import train_test_split


# In[14]:


train,test =train_test_split(df)


# In[15]:


test.index.size


# In[30]:


# function for finding root mean square error
def find_error(df):
   
    error = 0
    for index, row in df.iterrows():
        
        value = calculate_rating(df,row['reviewerID'], row['asin'])
        error += (value - row['overall'])**2
        if index>2 :
            break;
        print(index)
    # Mean square error
    error = error/df.shape[0]
    # root mean square error
    return math.sqrt(error)


# In[31]:


print("MSE is ..............",find_error(test[0:1]));


# In[11]:


import math


# In[12]:





# In[35]:


x=test[0:2]


# In[36]:


error = 0
for index, row in x.iterrows():

    value = calculate_rating(df,row['reviewerID'], row['asin'])
    error += (value - row['overall'])**2
    if index>2 :
        break;
    print(index)
# Mean square error
error = error/df.shape[0]
# root mean square error
print( math.sqrt(error) )

