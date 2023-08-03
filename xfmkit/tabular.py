import pandas as pd
from tabulate import tabulate


IGNORE_LINES=['sum','Back','Compton','Mo','MoL']

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

    N_MAJORS=3

    labels = df.columns.values.tolist()

    major_elements = []

    df_=df.copy()

    for i in range(len(labels)):
        col_ = df_.iloc[i].sort_values(ascending=False)

        col_ = col_.drop(IGNORE_LINES)

        majors_ = col_.index[0:N_MAJORS].tolist()

        major_elements.append(majors_)

    return major_elements


def nestlist_as_str(nested_list: list):
    string_list = []
    for i in range(len(nested_list)):
        if i == 0:
            string_ = f"{i: >2} " + "N/A"
        else:
            string_ = f"{i: >2} " + ''.join(nested_list[i])

        string_list.append(string_)

    return string_list