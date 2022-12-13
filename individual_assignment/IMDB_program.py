from pyspark import SparkConf
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

BUCKET = "dmacademy-course-assets"
file_before = "vlerick/pre_release.csv"
file_after = "vlerick/after_release.csv"

config = {
    "spark.jars.python": "com.amazonaws.auth.instanceprofilecredentialsprovider",
    "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
}

config2 = {
    "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1"
}

conf = SparkConf().setAll(config2.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

df1 = spark.read.csv(f"s3a://{BUCKET}/{file_before}", header=True)
df2 = spark.read.csv(f"s3a://{BUCKET}/{file_after}", header=True)

before = df1.toPandas()
after = df2.toPandas()

#IMDB code


# Data Preparation and cleaning
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
print("hey im here")

df[num] = num_imputer.fit_transform(df[num])
print("hey im not here")

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

val_pred = pandas_to_spark(val_pred)
val_pred.write.json(f"s3a://{BUCKET}/vlerick/aleksandra_dziurdzia")
