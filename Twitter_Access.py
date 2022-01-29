import os
import tweepy as tw
import pandas as pd

# API Key: YQCDwLhzaCIm5HwO9kChzpAKM
# API Key Secret: UwkmNXjGy47LeyJQmhrj4JIuQc6JKTpX0cgPKxWYvlD0eZVhCJ
# Bearer Token: AAAAAAAAAAAAAAAAAAAAAM8WYQEAAAAAu5ELfNHjphvmRiVrL6Zgyx%2BtSbs%3DUk0GMZV5Vz2JKwSTpjBEdPoHWx6Wr9zDdjR4dQ7DxDjEZuJfHt

import requests
import os
import json
from datetime import datetime
import calendar
import threading
import time


# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
# bearer_token = os.environ.get("BEARER_TOKEN")
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAM8WYQEAAAAAu5ELfNHjphvmRiVrL6Zgyx%2BtSbs%3DUk0GMZV5Vz2JKwSTpjBEdPoHWx6Wr9zDdjR4dQ7DxDjEZuJfHt'

search_url = "https://api.twitter.com/2/tweets/search/all"

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
# https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all
query_params = {'query': '#vaccine',
                'max_results': '10',
                'start_time':'2020-01-28T16:28:44.099Z',
                'end_time':'2022-01-28T16:28:44.099Z',
                'expansions':'author_id',
                'tweet.fields':'author_id,created_at,geo,lang,public_metrics'
                }

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FullArchiveSearchPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.request("GET", search_url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def extract_nested_values(it):
    if isinstance(it, list):
        for sub_it in it:
            yield from extract_nested_values(sub_it)
    elif isinstance(it, dict):
        for value in it.values():
            yield from extract_nested_values(value)
    else:
        yield it

def main():
    start_time = calendar.timegm(datetime(2020,9,20,0,0).timetuple())
    end_time = calendar.timegm(datetime(2021,1,30,0,0).timetuple())
    retrieve_num = 10 # this is the number of the tweets we can retrieve per request, ranging from 10 to 500. We can request once per second
    df = pd.DataFrame()

    for t in range(start_time,end_time,3600*24):
        query_params['end_time']= datetime.utcfromtimestamp(t).isoformat('T')+"Z"
        query_params['max_results']=retrieve_num
        result = connect_to_endpoint(search_url, query_params)
        tweets= result['data']
        users = result['includes']['users']
        for i in range(retrieve_num):
            _dict = {**users[i],**tweets[i]} # A very unstable approach to merge two dictionary, the id field is overwritten, but it's what we want because id for users are author_id in tweets

            print(tweets[i]['author_id']==users[i]['id'])
        print(t)


if __name__ == "__main__":
    main()

