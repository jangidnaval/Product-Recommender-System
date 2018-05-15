
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame 
import nltk

from sklearn.model_selection import train_test_split
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


df=pd.read_csv('../csv/product_product.csv')


# In[3]:


df.describe()


# In[4]:


def min_reviews_req(df):
    x=df.describe()
    min_count=x.loc['50%','count']
    #considering top 50% products by reviews count......
    new_df=df[df['count']>=min_count]
    return new_df
    
    


# In[5]:


new_df=min_reviews_req(df)


# In[6]:


new_df.index.size


# In[7]:


new_df.head()


# In[8]:


new_df.describe()


# In[9]:


dfProductReview=new_df.groupby("asin", as_index=False).mean()


# In[10]:


dfProductReview.asin.unique().size


# In[11]:


ProductReviewSummary = new_df.groupby("asin")["summary"].apply(list)
ProductReviewSummary = pd.DataFrame(ProductReviewSummary)
ProductReviewSummary.reset_index(inplace=True)


# In[12]:


ProductReviewSummary.head()


# In[13]:


ProductReviewSummary.to_csv("../csv/ProductReviewSummary.csv",index=False)


# In[14]:


ProductReviewSummary=pd.read_csv("../csv/ProductReviewSummary.csv")


# In[15]:


ProductReviewSummary.head()


# In[16]:


dfMerged=pd.merge(ProductReviewSummary,dfProductReview,on='asin')


# In[17]:


dfMerged.drop('count',inplace=True,axis=1)


# In[18]:


dfMerged.head()


# # Text Cleaning - Summary column

# In[19]:


#function for tokenizing summary
regEx = re.compile('[^a-z]+')
def cleanReviews(reviewText):
    reviewText = str.lower(reviewText)
    reviewText = regEx.sub(' ', reviewText).strip()
    return reviewText


# In[20]:


dfMerged.head()


# In[21]:


#reset index and drop duplicate rows
dfMerged['clean_summary'] = dfMerged['summary'].apply(lambda x:cleanReviews(x))
#dfMerged = dfMerged.drop_duplicates(['overall'], keep='last')
#dfMerged = dfMerged.reset_index()


# In[22]:


dfMerged.index.size


# In[23]:


dfMerged.head()


# In[24]:


reviews = dfMerged["clean_summary"] 
countVector = CountVectorizer(max_features = 400, stop_words='english') 
transformedReviews = countVector.fit_transform(reviews) 


# In[25]:


transformedReviews.A
#print(transformedReviews)
dfReviews = DataFrame(transformedReviews.A, columns=countVector.get_feature_names())
dfReviews = dfReviews.astype(int)


# In[26]:


product_id=pd.DataFrame()
product_id[['asin','overall']]=dfMerged[['asin','overall']]


# In[27]:


product_id.to_csv("../csv/product_id.csv",index=False,sep=',')


# In[28]:


product_id=pd.read_csv("../csv/product_id.csv")


# In[29]:


product_id.head()


# In[30]:


dfReviews=product_id.join(dfReviews)


# In[31]:


#save 
dfReviews.to_csv("../csv/Summary_Feature.csv",index=False)


# In[32]:


dfReviews=pd.read_csv("../csv/Summary_Feature.csv")


# In[33]:


#test train spilt.....
train,test=train_test_split(dfReviews,test_size=0.3)
print("Train sample size ",train.index.size)
print("Test sample size ",test.index.size)

train_features=train.drop(['asin','overall'],axis=1)
test_features=test.drop(['asin','overall'],axis=1)
           
train_array=np.array(train_features)
test_array=np.array(test_features)


# In[34]:


train.to_csv('../csv/train.csv',index=False)
test.to_csv('../csv/test.csv',index=False)


# In[35]:


neighbor = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
neighbor.fit(train.drop(['asin','overall'],axis=1))
# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
distances, indices = neighbor.kneighbors(train_array)


# In[36]:


test.head()


# In[551]:


product_id.iloc[0]['asin']


# In[552]:


test.asin.iloc[1]


# In[554]:


#find most related products
for i in range(test.index.size):
    a = neighbor.kneighbors([test_array[i]])
    #print(a)
    print(a[0])
    print(a[1])
    related_product_list = a[1]
    
    first_related_product = [item[0] for item in related_product_list]
    first_related_product = str(first_related_product).strip('[]')
    first_related_product = int(first_related_product)
                             
    second_related_product = [item[1] for item in related_product_list]
    second_related_product = str(second_related_product).strip('[]')
    second_related_product = int(second_related_product)
    
    third_related_product = [item[2] for item in related_product_list]
    third_related_product = str(third_related_product).strip('[]')
    third_related_product = int(third_related_product)
    
    test_product=test.asin.iloc[i]
    s1=product_id['asin'][first_related_product]
    s2=product_id['asin'][second_related_product]
    s3=product_id['asin'][third_related_product]
    

    print ("Based on product reviews, for ",test_product," average rating is ",test.overall.iloc[i])
    print ("The first similar product is ", s1 ," average rating is ",product_id["overall"][first_related_product])
    print ("The second similar product is ",s2 ," average rating is ",product_id["overall"][second_related_product])
    print ("The third similar product is ",s3 ," average rating is ",product_id["overall"][third_related_product])
    print ("-----------------------------------------------------------")
    


# In[564]:


from sklearn.metrics import confusion_matrix


# In[38]:


chk=pd.DataFrame(data={'acc':np.zeros(50),'n':np.zeros(50)})

train_rating=train['overall'].astype(int)
test_rating=test['overall'].astype(int)

for i in range (1,50,2):

    knnclf = neighbors.KNeighborsClassifier(i, weights='distance',algorithm='ball_tree')
    knnclf.fit(train_array, train_rating)


    predctions = knnclf.predict(test_array)
    #print(knnpreds_test)

    #print("Classification Report ...")
    #print(classification_report(test_rating,predctions))

    #print("Confussion Matrix....")
    #print(confusion_matrix(test_rating,predctions))
    
    chk['n'][i]=i
    chk['acc'][i]=accuracy_score(test_rating, predctions)
    
    print("n_neighbors  ",i)
    print("Accuracy:    ",accuracy_score(test_rating, predctions))
    print("MSE:         ",mean_squared_error(test_rating, predctions))
    print("\n")

