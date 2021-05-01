import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import re
import time
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Graduation data
GRAD_COHORT_OUTCOMES_DATA_2010 = "data/2005-2010_Graduation_Outcomes_-_School_Level.csv"

COHORT_COL = "Cohort Year"
GRAD_YR_COL = "Grad Year"
TOTAL_GRAD_N_COL = "Total Grads #"
TOTAL_GRAD_PER_COL = "Total Grads %"
TOTAL_DROPOUT_PER_COL = "Dropout %"

DBN_COL = "DBN"
DEMOGRAPHIC_COL = "Demographic"
TOTAL_COHORT_VAL = "Total Cohort"

# SAT Data Important Values
SAT_2010_DATA_PATH = "data\SAT__College_Board__2010_School_Level_Results.csv"
SAT_2012_DATA_PATH = "data\SAT_Results_2012.csv"
MAX_SAT_SCORE = 1600
EXCELLENT_SCORE_THRESHOLD = 1240
DECENT_SCORE_THRESHOLD = 1050

CR_COLUMN = "Critical Reading Mean"
MTH_COLUMN = "Mathematics Mean"
WRTNG_COLUMN = "Writing Mean"
DBN_COLUMN = "DBN"

# Location Data
LOCATION_DATA = "data\School_Locations_2019-2020.csv"
SCHL_SYS_COL = "system_code"
NTA_DIS_COL = "NTA_Name"
LAT_COL = "LATITUDE"
LONG_COL = "LONGITUDE"
SCHOOL_BBL = "Borough_block_lot"

# Healthy Food location Data
HEALTHY_FOOD_LOCATION_DATA = "data\Recognized_Shop_Healthy_Stores.csv"
HEALTHY_FOOD_LAT_COL = "Latitude"
HEALTHY_FOOD_LONG_COL = "Longitude"
HEALTHY_FOOD_STORE_NAME_COL = "Store Name"

# Distance to Healthy Food Location
DISTANCE_FROM_HEALTHY_FOOD = "Distance From HFL"
CLOSEST_HEALTHY_FOOD_VENDOR = "Closest HFV"

# Housing Data
HOUSING_DATA = "data\Housing_New_York_Units_by_Building.csv"
BBL_COL = "BBL"
ELIU_COL = "Extremely Low Income Units"
VLIU_COL = "Very Low Income Units"
LIU_COL = "Low Income Units"
HOUSING_LAT = "Latitude"
HOUSING_LONG = "Longitude"

LIU_DISTANCE = "Closest LIU Distance"
LIU_NEARBY_COL = "Num of LIU Near"

def main():

    grad_data = read_GRAD_data(
        data_set=GRAD_COHORT_OUTCOMES_DATA_2010,
        cohort_col=COHORT_COL,
        total_grad_col=TOTAL_GRAD_N_COL
    )

    grad_data = grad_data[grad_data[DEMOGRAPHIC_COL] == "Total Cohort"]

    # clean the location data's DBN System Code column for later
    location_data = pd.read_csv(LOCATION_DATA)
    location_data[SCHL_SYS_COL] = location_data[SCHL_SYS_COL].str.strip()
    
    # Combine the school's location data with the school's grad data
    school_data = grad_data.merge(location_data, left_on=DBN_COL, right_on=SCHL_SYS_COL)

    # Read the Healthy Food Vendor data
    hfv_data = pd.read_csv(HEALTHY_FOOD_LOCATION_DATA)

    # Find the closest HFV
    hfv_data = find_closest_hfv(
        school_data = school_data, 
        hfv_data = hfv_data
    )

    housing_data = pd.read_csv(HOUSING_DATA)
    nearby_LIU_data = find_number_of_low_income_house(
        school_data = school_data,
        housing_data = housing_data
    )

    data = school_data.merge(nearby_LIU_data, left_on=DBN_COL, right_on=DBN_COL)
    data = data.merge(hfv_data, left_on=DBN_COL, right_on=DBN_COL)

    data = gen_corr_matrix(data=data)

    data.to_csv("combined_data.csv")

"""
Name: read_GRAD_data

Purpose: Read the data regarding the graduate information for all NYC schools

Input: data_set - The data set to analyze
       cohort_col - The column containing the cohort data
       total_grad_col - The total graduate column

Output:
"""
def read_GRAD_data(data_set, cohort_col, total_grad_col):
    grad_data = pd.read_csv(data_set)
    
    # Filter the data to remove highschools who did not report one or more scoring statistics
    grad_data = grad_data.dropna()

    # Remove any Cohorts containing Aug (ie Aug 2006) or have a s in the data fields
    for index, row in grad_data.iterrows():
        if type(row[cohort_col]) is str and "Aug" in row[cohort_col]:
            grad_data = grad_data.drop(index)
            continue
        elif "s" in row[total_grad_col]:
            grad_data = grad_data.drop(index)
            continue
        
        """
        if "4 year June" not in row["Cohort"]:
            grad_data = grad_data.drop(index)
            continue
        """

    # insert grad year column
    grad_year = []
    for index, row in grad_data.iterrows():
        grad_year.append(int(row[cohort_col]) + 4)

    grad_data[GRAD_YR_COL] = grad_year

    return grad_data

"""
Name: find_closest_hfv

Purpose: Find the closest Healthy Food Vendor to each school

Input: school_data - The school data which includes the school's coordinates
       hfv_data - The healthy food vendor data I was able to acquire

Output: Data correlating the healthy food vendors closest to each school
"""
def find_closest_hfv(school_data, hfv_data):

    DBNS = []
    closest_hfv = []
    distances_from_hfv = []

    # Drop the repeats of schools to help thin out the data and improve speed
    data = school_data.drop_duplicates(subset=DBN_COL, keep="first")

    for school_idx, school_row in data.iterrows():

        school_lat = school_row[LAT_COL]
        school_long = school_row[LONG_COL]

        # Get the school's coordinated
        school_gp = (school_lat, school_long)

        closest_distance_from_hfv = 10
        closest_hfv_name = ""

        for hfv_idx, hfv_row in hfv_data.iterrows():

            # Get the healthy food vendor's coordinates    
            hfv_gp = (hfv_row[HEALTHY_FOOD_LAT_COL], hfv_row[HEALTHY_FOOD_LONG_COL])
            if math.isnan(hfv_row[HEALTHY_FOOD_LAT_COL]) or math.isnan(hfv_row[HEALTHY_FOOD_LONG_COL]):
                continue

            # Compute the distance between a school and a HFV in miles
            distance_from_hfv = geodesic(school_gp, hfv_gp).miles

            # Find the closest HFV
            if (distance_from_hfv < closest_distance_from_hfv):
                closest_distance_from_hfv = distance_from_hfv
                closest_hfv_name = hfv_row[HEALTHY_FOOD_STORE_NAME_COL]
    
        # record data for the closest HFV for the school
        DBNS.append(school_row[DBN_COL])
        distances_from_hfv.append(closest_distance_from_hfv)
        closest_hfv.append(closest_hfv_name)

    # Store the HFV data and computations in a data frame
    HFV_data =  {
        DISTANCE_FROM_HEALTHY_FOOD : distances_from_hfv,
        CLOSEST_HEALTHY_FOOD_VENDOR : closest_hfv,
        DBN_COL : DBNS
    }
    HFV_data_df = pd.DataFrame(HFV_data)

    return HFV_data_df

"""
Name: find_number_of_low_income_house

Purpose: Find the closests LIU near the school as well as the number of LIU's near the school

Input: school_data - The school data which includes the school's coordinates
       housing_data - The housing data I found

Output: A dataframe containing the information regarding the school's and LIU's near the schools
"""
def find_number_of_low_income_house(school_data, housing_data):

    DBNS = []
    distances_from_LIU = []
    number_of_low_income_units_near = []

    # Drop the repeats of schools to help thin out the data and improve speed
    data = school_data.drop_duplicates(subset=DBN_COL, keep="first")

    for school_idx, school_row in data.iterrows():

        school_lat = school_row[LAT_COL]
        school_long = school_row[LONG_COL]

        school_gp = (school_lat, school_long)

        school_boruogh_num = int(str(school_row[SCHOOL_BBL])[:1])

        num_of_units_near = 0

        closest_distance_from_LIU = 10

        for LIH_idx, LIH_row in housing_data.iterrows():
            
            if (math.isnan(LIH_row[BBL_COL])):
                continue

            if (int(str(LIH_row[BBL_COL])[:1]) != school_boruogh_num):
                continue
            
            LIH_gp = (LIH_row[HOUSING_LAT], LIH_row[HOUSING_LONG])
            if math.isnan(LIH_row[HOUSING_LAT]) or math.isnan(LIH_row[HOUSING_LONG]):
                continue

            # Compute the distance between a school and a HFV in miles
            distance_from_LIH = geodesic(school_gp, LIH_gp).miles

            if (distance_from_LIH < closest_distance_from_LIU):
                closest_distance_from_LIU = distance_from_LIH

            if (distance_from_LIH < .5):
                #num_of_units_near += 1
                num_of_units_near += LIH_row[ELIU_COL] + LIH_row[VLIU_COL] + LIH_row[LIU_COL]
        
        distances_from_LIU.append(closest_distance_from_LIU)
        number_of_low_income_units_near.append(num_of_units_near)
        DBNS.append(school_row[DBN_COL])

    num_of_LIUs_data =  {
        LIU_DISTANCE : distances_from_LIU,
        LIU_NEARBY_COL : number_of_low_income_units_near,
        DBN_COL : DBNS
    }
    num_of_LIUs_df = pd.DataFrame(num_of_LIUs_data)

    return num_of_LIUs_df

"""
Name: gen_corr_matrix

Purpose: Generate correlation matricies based on the data

Input: data - Our combined data

Output: None

Side-Effect: Creates 5 correlation matricies and saves their figures
"""
def gen_corr_matrix(data):
    
    boroughs_list = ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]
    boroughs = {
        "Manhattan" : 1,
        "Bronx" : 2,
        "Brooklyn" : 3,
        "Queens" : 4,
        "Staten Island" : 5
    }

    borough_column = []

    # Create a borough column
    for _, row in data.iterrows():
        borough_column.append(boroughs_list[int(str(row[SCHOOL_BBL])[:1]) - 1])
    
    data["borough"] = borough_column

    # Separate the data through the boroughs and generate correlation matricies based on the borough
    for key, _ in boroughs.items():
        
        borough_data = data[data["borough"] == key]

        interested_data = borough_data[["Total Grads #","Dropout #", "Grad Year", LIU_NEARBY_COL, LIU_DISTANCE, DISTANCE_FROM_HEALTHY_FOOD, TOTAL_GRAD_PER_COL, TOTAL_DROPOUT_PER_COL]]
        interested_data["Dropout #"] = interested_data["Dropout #"].astype('int64')
        interested_data["Total Grads #"] = interested_data["Total Grads #"].astype('int64')
        interested_data[TOTAL_GRAD_PER_COL] = interested_data[TOTAL_GRAD_PER_COL].astype('int64')
        interested_data[TOTAL_DROPOUT_PER_COL] = interested_data[TOTAL_DROPOUT_PER_COL].astype('int64')
        corrMatrix = interested_data.corr()

        sn.heatmap(corrMatrix, annot=True)
        plt.tight_layout()
        plt.title(f"{key} Correlation Matrix" )
        plt.savefig(f"{key}_Correlation.png")
        plt.clf()
    
    return data

########################### Useful functions for other data not being utilized ############################################
"""
Name: read_SAT_data

Purpose: Reads the SAT data and drops any NA rows

Input: data_set - The data set 

Output: The filtered SAT data set
"""
def read_SAT_data(data_set):
    sat_data = pd.read_csv(data_set)
    
    # Filter the data to remove highschools who did not report one or more scoring statistics
    sat_data = sat_data.dropna()

    sat_data = categorize_schools_via_SAT_scores(sat_data=sat_data)

    #plot_SAT_scores_by_cluster(sat_data= sat_data)

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
        
        total_score = 0
        if row[CR_COLUMN] is not "s":
            total_score = int(row[CR_COLUMN]) + int(row[MTH_COLUMN]) + int(row[WRTNG_COLUMN])
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

"""
Name: plot_SAT_scores_by_cluster

Purpose: Plot the SAT data total scores by class

Input: sat_data - The SAT data

Output: None

Side-Effects: Displays a plot showing the SAT data scores by cluster
"""
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
    plt.clf()

if __name__ == "__main__":
    main()