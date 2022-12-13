import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.linear_model import Lasso

BUCKET = "dmacademy-course-assets"
df_pre = "vlerick/pre_release.csv"
df_after = "vlerick/after_release.csv"

if len(os.environ.get("AWS_SECRET_ACCESS_KEY")) < 1:

    config = {"spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
          "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.f3.s3a.InstanceProfileCredentialsProvider"
    }
else:
    config = {"spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1"
    }


conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

df1 = spark.read.csv(f"s3a://{BUCKET}/{df_pre}", header=True)
df2 = spark.read.csv(f"s3a://{BUCKET}/{df_after}", header=True)

before = df1.toPandas()
after = df2.toPandas()

#IMDB code

# merge the 2 dataframes
# we merge on movie title as this is the only common variable in the two datasets
df = pd.merge(before, after,how='inner', on='movie_title') 
# we drop the variables we don't need
df = df.drop(columns = ["gross","num_critic_for_reviews","num_voted_users","num_user_for_reviews","movie_facebook_likes","director_name","actor_1_name","actor_2_name","actor_3_name","movie_title","actor_1_facebook_likes","actor_2_facebook_likes","actor_3_facebook_likes"])
# drop content rating
df = df.drop("content_rating", axis='columns')
# we delete all rows with missing values
df = df.dropna()
# we delete the duplicates and immediately update the dataframe
df.drop_duplicates(inplace = True)
# we see that the 22 duplicates have been removed since the number of observations decreased to 1044.
# we now replace all languages that are not English, French or Spanish with 'other language'
vals = df["language"].value_counts()[:3].index
df['language'] = df.language.where(df.language.isin(vals), 'other_language')
# we use the OneHotEncoder function to encode the language variable into dummies
# we turn language into dummies and check if it worked
ohe = OneHotEncoder()
df_1 = ohe.fit_transform(df[['language']])
df[ohe.categories_[0]] = df_1.toarray()
# we repeat this process for country
vals = df["country"].value_counts()[:6].index
df['country'] = df.country.where(df.country.isin(vals), 'other_country')
# we use OneHotEncoder again
ohe = OneHotEncoder()
df_2 = ohe.fit_transform(df[['country']])
df[ohe.categories_[0]] = df_2.toarray()
# we extract dummies from the genres column, by separating different string first, then we combine the newly
# created dummies and concatenate them with original dataset
df_dumm = df['genres'].str.get_dummies(sep = '|')
comb = [df, df_dumm]
df = pd.concat(comb, axis = 1)
# sum all genres except for the most common ones
df["other_genres"] = df["Animation"]+df["Biography"]+df["Documentary"]+df["Film-Noir"]+df["History"]+df["Music"]+df["Musical"]+df["Mystery"]+df["Sci-Fi"]+df["Short"]+df["Sport"]+df["War"]+df["Western"]
# now we replace their values by 1
df=df.replace(2,1)
df=df.replace(3,1)
# now we can delete the non-common genres
df = df.drop(columns = ["Animation","Biography","Documentary","Film-Noir","History","Music","Musical","Mystery","Sci-Fi","Short","Sport","War","Western"])
df = df.drop(columns=["genres","language","country"])
# But before we can start building models, we first have to extract the target variable and 
# the explanatory variables
# for multicollinearity reasons, we also drop the country variables, as they might be correlated with language.
x = df.drop(columns = ["imdb_score","USA","UK","France","Canada","Germany","Australia","other_country"])
y = df["imdb_score"]
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size = 0.25,random_state=42)

def accuracy_cont(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))
    accuracy = 1-mape
    return accuracy
# train model
rf = RandomForestRegressor(max_depth=10, min_samples_leaf =1, random_state=0)
rf.fit(X_train, y_train)
#predict regression forest 
array_pred = np.round(rf.predict(X_test),0)
#add prediction to data frame
y_pred = pd.DataFrame({"y_pred": array_pred},index=X_test.index)
val_pred = pd.concat([y_test,y_pred,X_test],axis=1)
val_pred
#Evaluate model
#by comparing actual and predicted value 
act_value = val_pred["imdb_score"]
pred_value = val_pred["y_pred"]

val_pred = pd.concat([pd.DataFrame(y_test),pred_value, pd.DataFrame(X_test)],axis=1)
print("Still working here12")
predictions_spark = spark.createDataFrame(val_pred)
predictions_spark.write.json(f"s3a://{BUCKET}/vlerick/aleksandra_dziurdzia")
