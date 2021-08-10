#!/usr/bin/env python
# coding: utf-8

# below is from https://medium.com/analytics-vidhya/exploratory-data-analysis-for-beginner-7488d587f1ec

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="ticks")
# style = 'dark','darkgrid','whitegrid' are some other styles
filename = 'COMPLETEhydrated.csv'
directory = os.path.join('data',filename)
hydrated = pd.read_csv(directory, dtype='unicode')

# We need to convert all columns into the right data type, maybe remove some unnecessary columns
hydrated["retweet_count"] = pd.to_numeric(hydrated["retweet_count"], downcast="float")
hydrated["favorite_count"] = pd.to_numeric(hydrated["favorite_count"], downcast="float")
hydrated["user_listed_count"] = pd.to_numeric(hydrated["user_listed_count"], downcast="float")
hydrated["user_statuses_count"] = pd.to_numeric(hydrated["user_statuses _count"], downcast="float")
hydrated = hydrated.drop(columns=["user_statuses _count"])
hydrated["user_favourites_count"] = pd.to_numeric(hydrated["user_favourites_count"], downcast="float")
hydrated["created_at"] = pd.to_datetime(hydrated["created_at"])

# Clean the dataframe
testdf = hydrated
testdf = testdf.drop(columns=['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1','id_str','from_user_id_str'])
testdf = testdf.drop(index=testdf[testdf.user_lang.notna()].index)
testdf = testdf.drop(columns=['user_lang'])

# Tweeter counts over time
testdf = testdf.sort_values(by=['created_at'])
weekly_tweet = testdf.resample('w',on='created_at').count()
weekly_tweet.index = weekly_tweet.index.date

ax = weekly_tweet.plot(kind='bar',y='letter_id_str',figsize=(8,5))
ax.set_xlabel('Date - Week Starting')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Establish a dataframe based on users
df_by_user = testdf.groupby('from_user')
user_dict = {}
for username, sub_df in df_by_user:
    sub_df.sort_values('created_at')
    data = {}
    data["tweets_count"] = sub_df["text"].count()
    data["user_created_at"] = pd.to_datetime(sub_df["user_created_at"][sub_df["user_created_at"].index[0]]).date()
    data["user_verified"] = sub_df["user_verified"].notnull().any()
    user_dict[username]=data

new_df = pd.DataFrame.from_dict(user_dict,'index')
new_df.to_csv(os.path.join("data","user_stats.csv"))
new_df = new_df.sort_values('tweets_count',ascending=False)
new_df = new_df.sort_values('user_created_at')
new_df = new_df.sort_values("user_verified")
new_df.user_created_at = pd.DatetimeIndex(new_df.user_created_at).to_period('M')
month_group = new_df.groupby('user_created_at')

for month,sub_df in month_group:
    print(month," --> ",sub_df.user_verified.value_counts(normalize=True)*100)

new_df.value_counts('user_verified')

# To investigate verified individuals:
# Step 1. Scrapping all user_description and make it a corpus.
# Step 2. Remove the stop words and count the TFs
# Step 3. Show stats of word count, choose top 50?
# Step 4. Arbitrarily define two brackets of words to distinguish between Media and Individuals
# Step 5. Run it through all verified users

df_verified = testdf[testdf.user_verified == 'TRUE']
user_des_list = df_verified.user_description.unique().tolist()
# Test Field on a single user (realBenTalks)
ben = testdf[testdf.from_user=='realBenTalks']
ben = ben.sort_values(by=['created_at'])

# A plot on how many tweets Ben tweeted each week:
ben = ben.set_index('created_at')
weekly_tweet = ben.resample('w').count().reset_index()
weekly_tweet.created_at = weekly_tweet.created_at.dt.date

ax = weekly_tweet.plot(kind='bar',x='created_at',y='letter_id_str',figsize=(8,5))
ax.set_xlabel('Date - Week Starting')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Here .astype() will cast all df object as "true", instead we can use "==" to match if the content is true
# hydrated['user_verified'] = hydrated['user_verified'].astype('bool')
hydrated['user_verified'] = hydrated['user_verified'] == "True"

# Don't seem to need this, if we want the sum of the verified user we can count df directly
# print(hydrated.isnull().values.sum())
hydrated['user_verified'].value_counts()

# Test if users are always verified:
cnt_true = 0
cnt_false =0
cnt_mix = 0
error = 0
sumcount = -1
for index,data in hydrated.groupby(['from_user']):
    if sumcount ==0 :
        break
    sumcount-=1
    if data.user_verified.value_counts().count()>1:
        cnt_mix+=1
        #print(data.user_verified.value_counts())
       # print(data.user_verified.values)
    elif data.user_verified.values[0]== True:
        cnt_true+=1
       # print(data.user_verified.value_counts())
        #print(data.user_verified.values)
    else:
        cnt_false+=1

print({'Verified:':cnt_true,"Not Verified":cnt_false,"Verified in between:":cnt_mix,"Anormaly:":error})
# We can also group it by selecting id_str or letter_id_str to see how many unique entires
hydrated.groupby('user_verified')[['id_str', 'letter_id_str']].nunique()

# seaborn histogram 

# not really sure what categories will be useful/possible here
sns.distplot(hydrated['user_listed_count'], hist=True, kde=False, 
             bins=9, color='blue',
             hist_kws={'edgecolor': 'black'})
# Add labels
plt.title('User Listed?')
plt.xlabel('user_listed_count')
plt.ylabel('Count')


sns.scatterplot(x=np.linspace(1, 292271, num=292271), y=hydrated['user_verified'])
# (x=hydrated['user_verified'], y=hydrated['user_statuses_count'])

# Heat map pearson correlation matrix
corrmat = hydrated.corr()
f, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(corrmat, vmax=.8, square=True)

# Light color, i.e., see on the right, scale 0.8 is highly correlated,
# and darker color below or around -0.2 is not correlated.

# Your heatmap is correct, you just forgot to change the dataframe name from pottermerged --> hydrated
plt.figure(figsize=(30, 30))
plt.title('Pearson Correlation of Features', size=15)
colormap = sns.diverging_palette(10, 220, as_cmap=True)
sns.heatmap(hydrated.corr(),
            cmap=colormap,
            square=True,
            annot=True,
            linewidths=0.1, vmax=1.0, linecolor='white',
            annot_kws={'fontsize': 12})
plt.show()

# In above correlation matrix, we printed the number also so it will be easy for us
# to see which are highly correlated and value close to 1.00 is highly correlated.
