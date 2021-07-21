import os
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

def train_rf(path):
    directory = "saved_rf"
    parent_dir = "TLA\Analysis"
    p = os.path.join(parent_dir, directory)
    if os.path.isdir(p) == False:
        os.mkdir(p)
    directory = "saved_vec"
    parent_dir = "TLA\Analysis"
    p = os.path.join(parent_dir, directory)
    if os.path.isdir(p) == False:
        os.mkdir(p)
    df = pd.read_csv(path)
    train_doc,test_doc,train_labels,test_labels = train_test_split(df['content'].values,df['sentiment'].values,test_size=0.33, random_state=42)
    
    vectorizer = CountVectorizer(ngram_range=(1,4),analyzer='char',max_features=25000)
    vector = vectorizer.fit_transform(train_doc)
    train_df= pd.DataFrame(vector.toarray())
    
    vector_test = vectorizer.transform(test_doc)
    test_df = pd.DataFrame(vector_test.toarray())

    rf=RandomForestClassifier()
    rf.fit(train_df.values,train_labels)
    
    print(rf.score(test_df,test_labels))
    print(path)
    file="TLA\Analysis\saved_vec\{}.pkl".format(path[-6:-4])
    pickle.dump(vectorizer,open(file,"wb"))
    
    
    filename="TLA\Analysis\saved_rf\{}.pkl".format(path[-6:-4])
    pickle.dump(rf,open(filename,"wb"))

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--path', action='store', type=str)
    my_parser.add_argument('--train_all_datasets', action='store', type=bool, default=False)
    args = my_parser.parse_args()
    if args.train_all_datasets == False:
        train_rf(args.path)
    else:
        directory = "saved_rf"
        parent_dir = "TLA\Analysis"
        p = os.path.join(parent_dir, directory)
        if os.path.isdir(p) == False:
            os.mkdir(p)
        directory = "saved_vec"
        parent_dir = "TLA\Analysis"
        p = os.path.join(parent_dir, directory)
        if os.path.isdir(p) == False:
            os.mkdir(p)
        for file in os.listdir("TLA\Datasets"):
            try:
                path = os.path.join("TLA\Datasets", file)
                df = pd.read_csv(path)
                train_doc, test_doc, train_labels, test_labels = train_test_split(df['content'].values,
                                                                                  df['sentiment'].values, test_size=0.33,
                                                                                  random_state=42)

                vectorizer = CountVectorizer(ngram_range=(1, 4), analyzer='char', max_features=25000)
                vector = vectorizer.fit_transform(train_doc)
                train_df = pd.DataFrame(vector.toarray())

                vector_test = vectorizer.transform(test_doc)
                test_df = pd.DataFrame(vector_test.toarray())

                rf = RandomForestClassifier()
                rf.fit(train_df.values, train_labels)

                print(rf.score(test_df, test_labels))

                file = "TLA\Analysis\saved_vec\{}.pkl".format(path[-6:-4])
                pickle.dump(vectorizer, open(file, "wb"))

                filename = "TLA\Analysis\saved_rf\{}.pkl".format(path[-6:-4])
                pickle.dump(rf, open(filename, "wb"))
            except:
                pass
