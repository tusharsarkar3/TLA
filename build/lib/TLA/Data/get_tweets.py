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
from distutils.sysconfig import get_python_lib

def get_data_for_lang(language):
    if language == None:
        lang_list = ['sv', 'th', 'nl', 'ja', 'tr', 'ur', 'id', 'pt', 'fr', 'zh-cn', 'ko', 'hi', 'es', 'fa', 'ro', 'ru',
                     'en']
    else:
        lang_list = [language]
    df_dict={}
    # sv-Swedish
    # th-Thai
    # nl-Dutch
    # ja-Japanese
    # tr-Turkish
    # ur-Urdu
    # id-Indonesian
    # pt-Portuguese
    # fr-French
    # zh-cn-Chinese Simplified
    # ko-Korean
    # hi-Hindi
    # es-Spanish
    # fa-Persian
    # ro-Romanian
    # ru-Russian
    # en-English
    for i in range(len(lang_list)):
        lang=lang_list[i]
        scraped_tweets = sntwitter.TwitterSearchScraper('filter:has_engagement min_faves:10000 lang:{}'.format(lang)).get_items()

        # slicing the generator to keep only the first 100 tweets
        sliced_scraped_tweets = itertools.islice(scraped_tweets, 500)

        # convert to a DataFrame and keep only relevant columns
        df = pd.DataFrame(sliced_scraped_tweets)['content']
        path = parent_dir = get_python_lib() + "/TLA/Data" + "/datasets"
        df.to_csv(path + '/get_data_{}.csv'.format(lang), sep=',', index=False)
        df_dict[lang_list[i]]=df
    return df_dict

    
    
    
    
  