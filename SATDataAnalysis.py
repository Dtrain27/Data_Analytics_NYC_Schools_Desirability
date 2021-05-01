import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAT_DATA_PATH = "data\SAT__College_Board__2010_School_Level_Results.csv"
MAX_SAT_SCORE = 1600
EXCELLENT_SCORE_THRESHOLD = 1240
DECENT_SCORE_THRESHOLD = 1050

CR_COLUMN = "Critical Reading Mean"
MTH_COLUMN = "Mathematics Mean"
WRTNG_COLUMN = "Writing Mean"

def main():

    sat_data = read_SAT_data(data_set=SAT_DATA_PATH)
    print(sat_data)
    sat_data = categorize_schools_via_SAT_scores(sat_data=sat_data)

    print(sat_data)

def read_SAT_data(data_set):
    sat_data = pd.read_csv(data_set)
    
    # Filter the data to remove highschools who did not report one or more scoring statistics
    sat_data = sat_data.dropna()

    sat_data = categorize_schools_via_SAT_scores(sat_data=sat_data)

    plot_SAT_scores_by_cluster(sat_data= sat_data)
    return sat_data

"""
The following method is used to determine which schools should be categorized into
the appropriate SAT ranges

Class 1: Excellent Score (1600 - 1240 <=)
Class 2: Decent Score (1239 - 1050 <=)
Class 3: Poor Score (1049 and under)
"""
def categorize_schools_via_SAT_scores(sat_data):

    data_classes = []
    total_scores = []
    for index, row in sat_data.iterrows():
        
        total_score = row[CR_COLUMN] + row[MTH_COLUMN] + row[WRTNG_COLUMN]
        total_scores.append(total_score)

        if total_score in range(1880, 2400):
            data_classes.append(1)
        elif total_score in range(1620, 1879):
            data_classes.append(2)
        elif total_score in range(1479, 1619):
            data_classes.append(3)
        else:
            data_classes.append(4)

    sat_data["SAT Score Class"] = data_classes
    sat_data["Total Scores"] = total_scores
    return(sat_data)

def plot_SAT_scores_by_cluster(sat_data):
    
    for color, label in zip('bgrm', [1, 2, 3, 4]):
        row_indexes = []
        for index, row in sat_data.iterrows():
            if row["SAT Score Class"] == label:
                row_indexes.append(index)
        
        subset = sat_data[sat_data["SAT Score Class"] == label]
        plt.scatter(row_indexes, subset["Total Scores"], s=20, c=color, label=str(label))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()