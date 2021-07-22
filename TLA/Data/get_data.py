from TLA.Data.get_tweets import get_data_for_lang
from TLA.Data.Pre_Process_Tweets import pre_process_tweet
import os
import pandas as pd
import argparse
from distutils.sysconfig import get_python_lib


def store_data(language, process = False):
    """
    Cretaes a .csv file for the language specified.
    Can create processed datasets if process flag is set as True
    Input-> language - a striing specifying the language you want the tweets in
            process - A Boolean to specify pre-processing tweets.
    Output-> x - a dataframe containing extracted data.
    
    """
    directory = "datasets"
    parent_dir = get_python_lib() + "/TLA/Data"
    path = os.path.join(parent_dir, directory)
    if os.path.isdir(path) == False:
        os.mkdir(path)
    df_dict = get_data_for_lang(language)
    if process == True:
        for file in os.listdir(path):
            path = os.path.join("path", file)
            df = pd.read_csv(path)
            df_processed = pre_process_tweet(df)
            df_processed.to_csv(path, sep=',', index=False)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--lang', action='store', type=str)
    my_parser.add_argument('--process', action='store', type=bool)
    args = my_parser.parse_args()
    if args.process == None:
        store_data(args.lang)
    else:
        store_data(args.lang, args.process)
