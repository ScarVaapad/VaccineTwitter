#!/usr/bin/env python
# coding: utf-8

# below is from https://medium.com/analytics-vidhya/exploratory-data-analysis-for-beginner-7488d587f1ec

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import nltk
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import datetime
import iso639


sns.set(style="ticks")
# style = 'dark','darkgrid','whitegrid' are some other styles
filename = 'hydrated_clean.csv'
directory = os.path.join('data',filename)
hydrated = pd.read_csv(directory, dtype='unicode')

# Tweet counts over time and interactive VIS
hydrated = hydrated.sort_values(by=['created_at'])
hydrated['created_at']=pd.to_datetime(hydrated['created_at'])
weekly_tweet = hydrated.resample('w',on='created_at').count()
weekly_tweet.index = weekly_tweet.index.date

ax = weekly_tweet.plot(kind='bar',y='letter_id_str',figsize=(8,5))
ax.set_xlabel('Date - Week Starting')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

daily_tweet = hydrated.resample('d',on="created_at").count()

tweetNum = go.Scatter(
    x=daily_tweet.index.strftime("%Y-%m-%d").values,
    y=daily_tweet["text"].values,
    name = "Tweeter Counts",
    line = dict(color = '#7F7F7F'),
    opacity = 0.8
)

layout = dict(
    title='Tweet Counts over time',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=7,
                     label='1w',
                     step='day',
                     stepmode='backward'),
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)

fig = make_subplots(rows=1,cols=1)
fig.add_trace(tweetNum)
fig.update_layout(layout)
plotly.offline.plot(fig, filename='Tweet_Counts_Over_Time.html')

# Most retweeted --> Problem with retweet count the value in the sheet is not reliable, but counting directly is also not ideal since not all the tweets are included
# For now only using counting numbers
RT_hydrated = hydrated[hydrated['text'].str.startswith('RT')]
df_slice = hydrated[['from_user','created_at','retweet_count','tweet_text','tweet_hashtags','tweet_language']]
txt_stats = df_slice['tweet_text'].value_counts().head(200)

# Hashtag co-occurance
from ast import literal_eval
hydrated['tweet_hashtags']=hydrated['tweet_hashtags'].apply(literal_eval)
hashtag_stats = {}
stemmer = nltk.stem.porter.PorterStemmer()
for index,item in hydrated['tweet_hashtags'].iteritems():
    for hashtag in item:
        hashtag = re.sub(r'#','',hashtag).lower()
        hashtag = stemmer.stem(hashtag)
        if hashtag in hashtag_stats:
            hashtag_stats[hashtag]+=1
        else:
            hashtag_stats[hashtag]=1
hashtag_stats={k: v for k, v in sorted(hashtag_stats.items(), key=lambda item: item[1],reverse=True)}
first100_list = list(hashtag_stats.items())[:100]
_hashtag,_count = zip(*first100_list)
tweetHashtag = go.Bar(
    x=_hashtag,
    y=_count
)
fig = make_subplots(rows=2,cols=1)
fig.add_trace(tweetHashtag)
tweetHashtag2 = go.Bar(
    x=_hashtag[5:],
    y=_count[5:]
)
fig.add_trace(tweetHashtag2,row=2,col=1)
plotly.offline.plot(fig, filename='Tweet_Hashtag.html')

# Pie-chart of language use on tweets (all tweets)
x = []
y = []
for k,v in hydrated['tweet_language'].value_counts().iteritems():
    try:
        _k=iso639.to_name(k)
    except:
        _k=k
    x.append(_k)
    y.append(v)

fig = make_subplots(rows=2,cols=1)
fig.add_trace(go.Pie(name='Language Distribution',labels=x,values=y))
plotly.offline.plot(fig,filename='Language_Distribution.html')

# Establish a dataframe based on users
# df_by_user = hydrated.groupby('from_user')
# user_dict = {}
# for username, sub_df in df_by_user:
#     sub_df.sort_values('created_at')
#     data = {}
#     data["tweets_count"] = sub_df["text"].count()
#     data["user_created_at"] = pd.to_datetime(sub_df["user_created_at"][sub_df["user_created_at"].index[0]]).date()
#     data["user_verified"] = sub_df["user_verified"].notnull().any()
#     user_dict[username]=data
#
# new_df = pd.DataFrame.from_dict(user_dict,'index')
# new_df.to_csv(os.path.join("data","user_stats.csv"))
# new_df = new_df.sort_values('tweets_count',ascending=False)
# new_df = new_df.sort_values('user_created_at')
# new_df = new_df.sort_values("user_verified")
# new_df.user_created_at = pd.DatetimeIndex(new_df.user_created_at).to_period('M')
# month_group = new_df.groupby('user_created_at')
#
# for month,sub_df in month_group:
#     print(month," --> ",sub_df.user_verified.value_counts(normalize=True)*100)

# Loading and investigate Age of the accounts
user_df = pd.read_csv(os.path.join("data","user_stats.csv"))
user_df = user_df.sort_values('user_created_at')
user_df.user_created_at = pd.to_datetime(user_df['user_created_at'])
d1 = user_df[user_df.user_created_at<=datetime.datetime(2020,3,1)]['user_created_at'].count()
d2 = user_df[(user_df.user_created_at>datetime.datetime(2020,3,1)) & (user_df.user_created_at<=datetime.datetime(2020,6,1))]['user_created_at'].count()
d3 = user_df[(user_df.user_created_at>datetime.datetime(2020,6,1)) & (user_df.user_created_at<=datetime.datetime(2020,9,1))]['user_created_at'].count()
d4 = user_df[(user_df.user_created_at>datetime.datetime(2020,9,1)) & (user_df.user_created_at<=datetime.datetime(2020,11,4))]['user_created_at'].count()
d5 = user_df[user_df.user_created_at>datetime.datetime(2020,11,4)]['user_created_at'].count()
fig = make_subplots(rows=1,cols=2)
fig.add_trace(go.Bar(name='# of users registered in the dataset',x=['Before March','March to June','June to August','August to Election','Post Election'], y= [d1,d2,d3,d4,d5]))
text = "Users registered after March accounted for %.2f%% of the data" % (((d2+d3+d4+d5)/(d1+d2+d3+d4+d5)).round(4)*100)
fig.add_trace(go.Bar(name='# of users registered after March 2020',x=['March to June','June to August','August to Election','Post Election'], y= [d2,d3,d4,d5]),row=1,col=2)
fig['layout'].update(
    annotations=[
        dict(
            xref = 'x2',
            yref = 'y1',
            x=1.5,
            y=80000,
            text=text
        )
    ]
)
plotly.offline.plot(fig, filename='User Account Registrations Overtime.html')



#new_df.value_counts('user_verified')

# To investigate verified individuals:
# Step 1. Scrapping all user_description and make it a corpus.
# Step 2. Remove the stop words and count the TFs
# Step 3. Show stats of word count, choose top 50?
# Step 4. Arbitrarily define two brackets of words to distinguish between Media and Individuals
# Step 5. Run it through all verified users
corpus_user_description_list = hydrated.user_description.unique().tolist()



df_verified = hydrated[hydrated.user_verified == 'TRUE']
user_des_list = df_verified.user_description.unique().tolist()
# Test Field on a single user (realBenTalks)
ben = hydrated[hydrated.from_user=='realBenTalks']
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

# Languages used
lang_stats = hydrated['tweet_language'].value_counts()
minor = 0
lang_result = []
for index,value in lang_stats.iteritems():
    if value<500:
        minor+=value
    else:
        lang_result.append(value)
lang_result.append(minor)

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
