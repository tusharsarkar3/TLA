import os
import pickle
import pandas as pd
import numpy as np
import argparse

def emotion(tweet,lang_code):
    try:
        if isinstance(pd.read_csv(tweet),pd.DataFrame) == True:
            tweet = np.array(pd.read_csv(tweet))
    except:
        if isinstance(tweet,str) == True:
            tweet = np.array([tweet])
        else:
            return "First Argument must be of string or numpy array DataType"
    path_vec="TLA\Analysis\saved_vec"
    file_vec=os.path.join(path_vec,lang_code+".pkl")
    vec=pickle.load(open(file_vec,'rb'))
    
    path_rf="TLA\Analysis\saved_rf"
    file_rf=os.path.join(path_rf,lang_code+".pkl")
    model=pickle.load(open(file_rf,'rb'))
    v=vec.transform(tweet)
    test_df = pd.DataFrame(v.toarray())
    
    return model.predict(test_df)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--predict', action='store', type=str,required=True)
    my_parser.add_argument('--lang', action='store', type=str,required=True)
    args = my_parser.parse_args()
    prediction = emotion(args.predict,args.lang)
    print(prediction)
