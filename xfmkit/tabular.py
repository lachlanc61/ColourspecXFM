import pandas as pd
from tabulate import tabulate
import xfmkit.config as config

import logging
logger = logging.getLogger(__name__)

ignore_lines=config.get('element_lists', 'ignore_lines')

def get_df(classavg, labels):
    """
    create a dataframe from class averages and labels
    """

    df = pd.DataFrame(data=classavg, columns=labels)

    return df

def printout(df):
    """
    display a dataframe as table
    """
    print(tabulate(df, headers='keys', tablefmt='psql'))

    return

def get_major_list(df):

    class_majors = []

    df_=df.copy()

    for i in range(len(df.index)):
        row_ = df_.iloc[i].sort_values(ascending=False)

        #filter method not working as row_ is not a df anymore and thus does not have labels
        #use ignore instead of filter for now, TO-DO fix this
        drop_filter = row_.filter(ignore_lines)

        row_ = row_.drop(ignore_lines, errors='ignore')

        majors_=[]

        for i in range(len(row_.index)):
            if len(majors_) == 0 or (row_[i] > 50000):# and row_[i] > row_[majors_[-1]]/10): 
                    #majors_.append(row_.index[i])
                    try:
                        percent = int(round(row_[i]/10000,0))
                    except:
                        percent = 0

                    majors_.append(str(row_.index[i])+str(percent))

        class_majors.append(majors_)

    return class_majors


def nestlist_as_str(nested_list: list):
    string_list = []
    for i in range(len(nested_list)):
        if i == 0:
            string_ = f"{i: >2} " + "N/A"
        else:
            string_ = f"{i: >2} " + ' '.join(nested_list[i])

        string_list.append(string_)

    return string_list