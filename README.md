# Product Recommender System

This Project deals with identifying and predicting the most relevant products for
an user based on his/her previous interaction. We are interacting with the recom-
mender systems in our day-to-day life like product recommendation in e-commerce
sites (Amazon, Flipkart), friend recommendation in social networking sites (Face-
book, Instagram), movie and video recommendation in YouTube, Netflix and job
recommendation in Linkedin etc.

 Based on the research on some existing models and algorithms, we make application-
 specific three recommendation systems -

1. User-User Collaborative Filtering using Pearson similarity- In user based recom-
mender system, we look for users who share the same rating patterns with the active
user and use the ratings from those like- minded users to calculate a prediction for
the active user.

2. Item-Item Collaborative Filtering using K-nearest neighbours- In it, we build
an item-item matrix determining Relationship between pairs of item and infer the
tastes of the current user by examining the matrix and matching that user data.

3. Popularity Model- It recommends the most popular products rated by the users.


# Dataset
Dataset is taken from http://snap.stanford.edu/data/web-Amazon.html



References:
J. McAuley and J. Leskovec. Hidden factors and hidden topics: understanding rating dimensions with review text. RecSys, 2013.
http://jmcauley.ucsd.edu/data/amazon/

