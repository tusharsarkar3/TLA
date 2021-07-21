import os
import pandas as pd
import numpy as np
from TLA.Analysis.lang_mapping import mapping
from distutils.sysconfig import get_python_lib

# merge all data
def analysis_table2():
    lang_dict = mapping()
    df=pd.DataFrame( columns=["type","total_tweets","pos","neg","percentage_positive","percentage_negative"],dtype = float)
    directory = "analysis"
    parent_dir = get_python_lib() + "/TLA/Analysis"
    p = os.path.join(parent_dir, directory)
    if os.path.isdir(p) == False:
        os.mkdir(p)
    sumtot=0
    sumpostot=0
    sumnegtot=0
    for filename in os.listdir(get_python_lib() +"/TLA/Datasets"):
        sumpos=0
        sumneg=0
        f = os.path.join(get_python_lib() +"/TLA/Datasets", filename)
        df1=pd.read_csv(f)
        t_tweets=df1.shape[0]
        sumtot+=t_tweets
        for x in df1["sentiment"]:
            if x == "Positive":
                sumpos+=1

            else:
                sumneg+=1


        sumpostot+=sumpos
        sumnegtot+=sumneg

    pcentpostot=(sumpostot/sumtot)*100
    pcentnegtot=(sumnegtot/sumtot)*100

    df.loc[len(df.index)] = ['Total_tweets', sumtot,sumpostot,sumnegtot,pcentpostot,pcentnegtot]

    df.replace(lang_dict, inplace=True)
    df.to_csv(get_python_lib() + "/TLA/Analysis/analysis/table2.csv",index=False)

if __name__ == "__main__":
    analysis_table2()
            
            