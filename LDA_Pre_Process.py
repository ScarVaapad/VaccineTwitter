import gensim.models
import pandas as pd
import os
import re
#import gensim
from nltk.stem import WordNetLemmatizer
import emoji
import nltk
from langdetect import detect
import iso639

lemmatizer = WordNetLemmatizer()
__stop_words = set(nltk.corpus.stopwords.words('english'))

def mention_extraction(tweet):
    mention = re.findall("@\w+",tweet)
    res = ""
    for w in mention:
        res += w
        res +=','
    return res

def hashtag_extraction(tweet):
    hashtag = re.findall("#[A-Za-z0-9]+",tweet)
    res = ""
    for w in hashtag:
        res += w
        res +=','
    return res

def is_flag_emoji(c):
    return "\U0001F1E6\U0001F1E8" <= c <= "\U0001F1FF\U0001F1FC" or c in ["\U0001F3F4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f", "\U0001F3F4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f", "\U0001F3F4\U000e0067\U000e0062\U000e0077\U000e006c\U000e0073\U000e007f"]

def emoji_extraction(tweet):
    emj = ''.join(c for c in tweet if (c in emoji.UNICODE_EMOJI['en'] or is_flag_emoji(c)))
    return emj

def wordbag(tweet):
    tweet = re.sub("@\w+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = ''.join(c for c in tweet if (c not in emoji.UNICODE_EMOJI['en']) or ~is_flag_emoji(c)) #Remove Emojis
    tweet = tweet.replace("#", "") #Remove hashtag sign but keep the text
    tweet = tweet.replace(":","")
    tweet = tweet.lower()
    wordbag = ' '.join([lemmatizer.lemmatize(i, 'v')
              for i in tweet.split() if i not in __stop_words])
    return wordbag

def language(tweet):
    lang = ""
    try:
        lang = detect(tweet)
        try:
            lang = iso639.to_name(lang)
        except:
            lang = lang
    except:
        lang = "error"
    return lang


# df = pd.read_csv(os.path.join('data',"COMPLETEhydrated.csv"))

# #Clean unwanted fields
# df.drop(columns=['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1','id_str','from_user_id_str'],inplace=True)
# df.drop(index=df[df.user_lang.notna()].index,inplace=True)
# df.drop(columns=['user_lang'],inplace=True)
#
# total_row = len(df)
# count = 0
# for index,row in df.iterrows():
#     twt = row['text']
#     df.loc[index,'tweet_mentions'] = mention_extraction(twt)
#     df.loc[index,'tweet_emjs'] = emoji_extraction(twt)
#     df.loc[index,'tweet_hashtags'] = hashtag_extraction(twt)
#     df.loc[index,'tweet_language'] = language(twt)
#     df.loc[index,'tweet_wordbag'] = wordbag(twt)
#
#     count+=1
#     if(count%100 == 0):
#         print(str(count),'/',str(total_row),':',str(count/total_row*100)+'% done')
#
# df.to_csv(os.path.join('data','hydrated_clean_1.csv'))
