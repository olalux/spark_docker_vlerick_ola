#!/usr/bin/env python
# coding: utf-8

# # Data exploration
# 
# First I load the data and check what it looks like.

# In[2095]:


get_ipython().system('pip install xgboost')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


# In[2096]:


#Loading the pre-release data file and taking a look at it
before = pd.read_csv('../data/pre_release.csv')
before.head()


# In[2097]:


before.shape


# In[2098]:


before.describe(include='object')


# In[2099]:


#Loading the after-release data file
after = pd.read_csv('../data/after_release.csv')
after.head()


# In[2100]:


after.shape


# In[2101]:


after.describe(include='object')


# # Data preparation & cleaning
# After looking at my data I decided that there are three options for what I can predict:
# 
#     1. IMDB score
#     2. Gross
#     3. Movie facebook likes
#     
# Hence, I decided to remove all other columns from the after release since I will not need them in my model. I decided to keep three potential target variables at this stage so that I can decide which one is best to predict after the data exploration.

# In[2102]:


after_n = after.drop(['num_critic_for_reviews', 'num_voted_users', 'num_user_for_reviews'], axis = 'columns')
after_n


# In[2103]:


#I merged the before release and after release dataframes 

df = before.merge(after_n,how="inner",on="movie_title")
df.head()


# In[2104]:


df.shape


# In[2105]:


#After merging I make sure that the same movies don't appear twice 
df = df.drop_duplicates('movie_title')
df.shape


# In[2106]:


#I delete the names of the director and actors because I assume that their facebook likes are more insightful
df = df.drop(['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name'], axis = 'columns')


# In[2107]:


df.isnull().sum()


# In[2109]:


#I group the categorical and numerical values into lists to help me replace the missing values

num = df.select_dtypes(include = 'float64').columns.tolist()
cat = df.select_dtypes(include = 'object').columns.tolist()
#I remove genres from the list because it has multiple values so I will deal with it later
cat.remove('genres')


# In[2110]:


from sklearn.impute import SimpleImputer
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat] = cat_imputer.fit_transform(df[cat])


# In[2111]:


num_imputer = SimpleImputer(strategy='median')
df[num] = num_imputer.fit_transform(df[num])


# In[2114]:


#I converted the integer types to float to prevent future errors
df["cast_total_facebook_likes"] = df["cast_total_facebook_likes"].astype(float)
df['cast_total_facebook_likes'].dtypes


# In[2118]:


#checking what languages are most frequent
df['language'].value_counts()


# In[2119]:


df['english_language'] = df.loc[df['language'] == 'English', 'language']
df ['english_language'] = df['english_language'].fillna (0)
df['english_language'] = df['english_language'].replace(['English'], 1 )
df = df.drop(['language'], axis = 'columns')
df['english_language'].value_counts()


# In[2121]:


df['european'] = 0
df.loc[df['country'].isin(['UK', 'France', 'Germany', 'Spain', 'Italy', 'Netherlands', 'Denmark', 'Iceland', 'Russia', 'Ireland', 'Czech Republic', 'Romania', 'Poland', 'Norway', 'Belgium', 'Sweden']), 'european'] = 1
df


# In[2122]:


df['USA'] = 0
df.loc[df['country'] == 'USA', 'USA'] = 1
df['Other_countries'] = 0
df.loc[(df['european'] == 0) & (df['USA'] == 0), 'Other_countries'] = 1
df = df.drop(['country'], axis = 'columns')
df


# In[ ]:





# In[2124]:


#I noticed that genres is has multiple values so I split it on delimeter and create a dummy
df = pd.concat([df, df['genres'].str.get_dummies('|')], axis=1)
df.drop('genres', axis = 1, inplace = True)
df


# In[2125]:


#I make a new group of categorical variables for the new column titles
cat = df.select_dtypes(include = 'object').columns.tolist()
cat


# In[2126]:


#I make dummy variables for the remaining categorical variables
for i in cat:
    df = pd.concat([df, df[i].str.get_dummies()], axis=1)
    df.drop([i], axis=1, inplace=True)


# # Modelling

# In[2127]:


#Splitting my dataframe into the training and test set, using a 1:4 ratio
X_train, X_test, y_train, y_test = train_test_split(df.drop(['imdb_score'], axis = 'columns'), df.imdb_score, test_size = 0.2)


# In[2128]:


#Training the model
model = RandomForestRegressor(random_state = 1)
model.fit(X_train, y_train)


# In[2129]:


#Use the RandomForestRegressor model to make a prediction

array_pred = np.round(model.predict(X_test),0)

y_pred = pd.DataFrame({"y_pred": array_pred},index=X_test.index) #index must be same as original database
val_pred = pd.concat([y_test,y_pred,X_test],axis=1)
val_pred


# In[ ]:




