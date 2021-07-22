import os
import pandas as pd
import snscrape
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm
import math
import snscrape.modules.twitter as sntwitter
import itertools

def remove_Punctuations(x):
    """ Removes Punctuations from a given string
        Input-> x - A string.
        output-> x - A string.
     """
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for letter in x:
        if letter not in punctuations:
            no_punct = no_punct + letter
    return no_punct.strip(" ")

def deEmojify(text):
    """ Removes Emojis from a given string
        Input-> x - A string.
        output-> x - A string.
     """
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


def remove_url(x):
     """ Removes Urls from a given string
        Input-> x - A string.
        output-> x - A string.
     """
    result = re.sub(r"http\S+", "", x)
    return result

def remove_everything(x):
     """ Removes remaining undesiarable characters from text.
        Input-> x - A string.
        output-> x - A string.
     """

    if "\n" in x:
        x=str(x.split("\n"))

    mn = re.sub("[^A-Za-z]", "", x)
    return mn


def clean_up(x):
    """ Implement pre-processing steps for a given string.
    Steps include:
    1> remove urls
    2> remove emojis
    3> remove punctuations
        Input-> x - A string.
        output-> x - A string.
     """
    
    x=remove_Punctuations(x)
    x=deEmojify(x)
    x=remove_url(x)
    x=remove_everything(x)
    return x





def pre_process_tweet(df):
     """ Implement pre-processing steps for given strings in a dataframe.
    Steps include:
    1> remove urls
    2> remove emojis
    3> remove punctuations
        Input-> x - A dataframe.
        output-> x - A dataframe.
     """
    
    all_tweets=[]
    for i in range (df.shape[0]):

        low=[]
        tweet=df.iloc[i,0]


        word_l=tweet.split(" ")
        for j in word_l:
                if "\n" in j:
                    xy=j.split("\n")
                    word_l.extend(xy)
                    word_l.remove(j)
        for w in word_l:
                x=clean_up(w)
                sw=[]
                for kjh in stopwords.fileids():
                    sw.extend(stopwords.words('{}'.format(kjh)))
                    
                    
                mn = word_tokenize(x)
                for t in mn:
                    if t.lower() not in sw:


                        low.append(t.lower())


        all_tweets.append(low)
    df["new"]=all_tweets
    return df



