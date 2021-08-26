#!/usr/bin/env python
# coding: utf-8

# below is from https://medium.com/analytics-vidhya/exploratory-data-analysis-for-beginner-7488d587f1ec

import pandas as pd
import numpy as np
import os
import re
import emoji
import nltk
from langdetect import detect

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
hydrated = hydrated.drop(columns=['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1','id_str','from_user_id_str'])
hydrated = hydrated.drop(index=hydrated[hydrated.user_lang.notna()].index)
hydrated = hydrated.drop(columns=['user_lang'])

#corpus_user_description_list = hydrated.user_description.unique().tolist()
corpus_tweet_list = hydrated.text.tolist()
# Clean Tweets, put different set of info into different columns
words = set(nltk.corpus.words.words())

def cleaner(tweet):
    mention = re.findall("@[A-Za-z0-9]+",tweet)
    hashtag = re.findall("#[A-Za-z0-9]+",tweet)
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    emj = ''.join(c for c in tweet if c in emoji.UNICODE_EMOJI['en']) #Extract Emoji
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI['en']) #Remove Emojis
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    #tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
                    # if w.lower() in words or not w.isalpha())
    return tweet,mention,emj,hashtag

tweets=[]
mentions=[]
emjs=[]
hashtags=[]
langs = []
count = 0
for twt in corpus_tweet_list:
    count+=1
    if(count%1000==0):
        print("beep")
    tweet,mention,emj,hashtag = cleaner(twt)

    tweets.append(tweet)
    mentions.append(mention)
    emjs.append(emj)
    hashtags.append(hashtag)
    lang = ""
    try:
        lang = detect(tweet)
    except:
        lang = "error"
    langs.append(lang)

hydrated['tweet_text']=tweets
hydrated['tweet_mentions']=mentions
hydrated['tweet_emojis']=emjs
hydrated['tweet_hashtags']=hashtags
hydrated['tweet_language']=langs

hydrated.to_csv(os.path.join('data','hydrated_clean.csv'))
