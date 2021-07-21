import torch
import os
import pandas as pd
import numpy as np
from lang_mapping import mapping

def analysis_table():
    lang_dict = mapping()
    directory = "analysis"
    parent_dir = "TLA\Analysis"
    p = os.path.join(parent_dir, directory)
    if os.path.isdir(p) == False:
        os.mkdir(p)

    df=pd.DataFrame( columns=["language","total_tweets","pos","neg","percentage_positive","percentage_negative"],dtype=float)

    for filename in os.listdir('TLA\Datasets'):
        sumpos=0
        sumneg=0
        f = os.path.join('TLA\Datasets', filename)
        df1=pd.read_csv(f)
        t_tweets=df1.shape[0]
        for x in df1["sentiment"]:
            if x == "Positive":
                sumpos+=1

            else:
                sumneg+=1
        pcentpos=(sumpos/t_tweets)*100
        pcentneg=(sumneg/t_tweets)*100

        df.loc[len(df.index)] = [filename[-6:-4], t_tweets,sumpos,sumneg,pcentpos,pcentneg]

    df.replace(lang_dict,inplace=True)
    df.to_csv(r'TLA\Analysis\analysis\table1.csv',index=False)

if __name__ == "__main__":
    analysis_table()