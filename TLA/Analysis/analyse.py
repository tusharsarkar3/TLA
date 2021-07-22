from TLA.Analysis.table_1 import analysis_table
from TLA.Analysis.table_2 import analysis_table2

def analyse_data():
    """
    Creates .csv files representing results of sentiment analysis carried for the specified dataset.
    Input-> Takes no input
    Output> .csv files.
    """
    analysis_table()
    analysis_table2()

if __name__ == "__main__":
    analysis_table()
    analysis_table2()
