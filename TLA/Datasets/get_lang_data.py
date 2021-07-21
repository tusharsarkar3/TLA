import pandas as pd
import argparse

def language_data(lang):
    path = "TLA\Datasets\get_data_" + str(lang) + ".csv"
    return pd.read_csv(path)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--lang', action='store', type=str,required=True)
    args = my_parser.parse_args()
    print(language_data(args.lang))