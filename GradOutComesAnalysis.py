import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRAD_COHORT_OUTCOMES_DATA = "data/2005-2010_Graduation_Outcomes_-_School_Level.csv"

COHORT_COL = "Cohort"
GRAD_YR_COL = "Grad Year"
TOTAL_GRAD_N_COL = "Total Grads - n"
def main():

    grad_data = read_GRAD_data(data_set=GRAD_COHORT_OUTCOMES_DATA)

    print(grad_data)

def read_GRAD_data(data_set):
    grad_data = pd.read_csv(data_set)
    
    # Filter the data to remove highschools who did not report one or more scoring statistics
    grad_data = grad_data.dropna()

    # Remove any Cohorts containing Aug (ie Aug 2006) or have a s in the data fields
    for index, row in grad_data.iterrows():
        if "Aug" in row[COHORT_COL]:
            grad_data = grad_data.drop(index)
            continue
        elif "s" in row[TOTAL_GRAD_N_COL]:
            grad_data = grad_data.drop(index)
            continue
    
    # insert grad year column
    grad_year = []
    for index, row in grad_data.iterrows():
        grad_year.append(int(row[COHORT_COL]) + 4)

    grad_data[GRAD_YR_COL] = grad_year

    return grad_data

if __name__ == "__main__":
    main()