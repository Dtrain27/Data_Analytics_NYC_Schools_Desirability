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
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

# Graduation data
GRAD_COHORT_OUTCOMES_DATA_2010 = "data/2005-2010_Graduation_Outcomes_-_School_Level.csv"
GRAD_COHORT_OUTCOMES_DATA = "data/2001_-_2013_Graduation_Outcomes.csv"

COHORT_COL = "Cohort Year"
GRAD_YR_COL = "Grad Year"
TOTAL_GRAD_N_COL = "Total Grads #"
TOTAL_GRAD_PER_COL = "Total Grads %"
TOTAL_DROPOUT_PER_COL = "Dropout %"
DROPOUT_NUM_COL =  "Dropout #"

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

# School Safety Reports
SAFET_REPORT_DATA = "School_Safety_Report_2010_-_2016.csv"

# Combined data columns
BOROUGH_COL = "borough"
TRUE_CLASS_COL = "class"

def main():
    data = pd.read_csv("combined_data.csv")

    data = filter_and_one_hot_encode_data(data = data)

    classifying_data = classify_data_by_drop_and_grad_rates(
        data = data
    )

    classify_using_KNN(data = classifying_data)

    classify_using_randomforest(data = classifying_data)

    classify_using_kmeans(data = classifying_data)

    classify_using_logistic_regression(data = classifying_data)

"""
Name: filter_and_one_hot_encode_data

Purpose: To one hot encode the borough column so they can be used in our models

Input: data - Our data to one hot encode

Output: Our data with the borough column one hot encoded
"""
def filter_and_one_hot_encode_data(data):
  # One hot encode the data for the borough
    data = data[[TOTAL_GRAD_N_COL, GRAD_YR_COL, LIU_NEARBY_COL, LIU_DISTANCE, DISTANCE_FROM_HEALTHY_FOOD, TOTAL_GRAD_PER_COL, TOTAL_DROPOUT_PER_COL, DROPOUT_NUM_COL, BOROUGH_COL]]
    one_hot = pd.get_dummies(data[BOROUGH_COL])
    
    # Drop column borough as it is now encoded
    data = data.drop(BOROUGH_COL,axis = 1)
    
    # Join the encoded df
    data = data.join(one_hot)

    return data

"""
Name: classify_data_by_drop_and_grad_rates

Purpose: Classify the data by drop and grad numbers/%

Input: data - Our data we wish to classify

Output: A modified form of the data containing a class column

        +1 - A good school to attend
        -1 - An undesirable school

        Note: We determine this through the comparison of grad #'s and grad % versus
              the drop out # and drop %
"""
def classify_data_by_drop_and_grad_rates(data):

    THRESHOLD_DROPOUT_PER = 10
    THRESHOLD_DROPOUT_NUM = 5

    THRESHOLD_GRAD_PER = 80

    n, d = data.shape

    classes = []
    for idx, row in data.iterrows():

        grad_num = row[TOTAL_GRAD_N_COL]
        grad_percent = row[TOTAL_GRAD_PER_COL]

        dropout_num = row[DROPOUT_NUM_COL]
        dropout_percent = row[TOTAL_DROPOUT_PER_COL]
        
        if dropout_percent > THRESHOLD_DROPOUT_PER and dropout_num > THRESHOLD_DROPOUT_NUM:
            classes.append(-1)
        else:
            classes.append(+1)

    data[TRUE_CLASS_COL] = classes

    plot_classes(
        data = data
    )

    return data

"""
Name: plot_classes

Purpose: Plot the data based on the class column

Input: data our data to plot

Output: None

Side-Effects: Saves a plot of the classified data
"""
def plot_classes(data):
    
    THRESHOLD_DROPOUT_PER = 10
    THRESHOLD_DROPOUT_NUM = 5

    drop_per_data = data[TOTAL_DROPOUT_PER_COL]
    drop_num_data = data[DROPOUT_NUM_COL]
    
    have_pos_label = False
    have_neg_label = False

    # Plot our data points based on their class 
    for drop_per, drop_num in zip(drop_per_data, drop_num_data):
        if drop_per > THRESHOLD_DROPOUT_PER and drop_num > THRESHOLD_DROPOUT_NUM:
            plt.plot(drop_per, drop_num, "r+", label="Not Desired school" if not have_pos_label else "")
            have_pos_label = True
        else:
            plt.plot(drop_per, drop_num, "bo", label="Desired school" if not have_neg_label else "")
            have_neg_label = True

    plt.xlabel("Dropout Percentage")
    plt.ylabel("Droput Number")
    plt.title("Dropout Percentage vs Droput Number")
    plt.legend()
    plt.savefig("Data Classification.png")
    plt.clf()

"""
Name: classify_using_KNN

Purpose: Classify our data points using KNN

Input: data - Our data to classify

Output: None

Side-Effects: Produces two graphs measuring the MSE and accuracy of the models
"""
def classify_using_KNN(data):

    n, d = data.shape

    #Separate the data into training and testing points
    train_num = int(n*.7)
    test_num = n-train_num

    train_data = data.iloc[:train_num,:]
    test_data = data.iloc[train_num:n,:]

    train_y_data = train_data[[TRUE_CLASS_COL]].to_numpy()
    train_x_data = train_data.drop(TRUE_CLASS_COL,axis = 1)

    test_y_data = test_data[[TRUE_CLASS_COL]].to_numpy().flatten()
    test_x_data = test_data.drop(TRUE_CLASS_COL,axis = 1)

    # Compute the KNN model and vary K between 1 through 50
    errors = []
    accuracys = []
    k_vals = []
    best_accuracy = 0
    for neighbors in range(2, 50):
        scaler = StandardScaler()
        scaler.fit(train_x_data)

        train_x_data = scaler.transform(train_x_data)
        test_x_data = scaler.transform(test_x_data)

        classifier = KNeighborsClassifier(n_neighbors=neighbors)
        classifier.fit(train_x_data, train_y_data.ravel())

        y_pred = classifier.predict(test_x_data)

        errors.append(np.mean(y_pred != test_y_data))

        report = classification_report(test_y_data, y_pred, output_dict=True)
        accuracys.append(report['accuracy'])
        if best_accuracy < report['accuracy']:
            best_accuracy = report['accuracy']
        k_vals.append(neighbors)

    # Plot our various k values and their error rates and accuracies
    plt.plot(
        range(2, 50), 
        errors, 
        color='black', 
        linestyle='dashed', 
        marker='o',
        markerfacecolor='blue', 
        markersize=10
    )
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')

    plt.savefig("Compare Error Rate KNN.png")
    plt.clf()

    plt.plot(
        range(2, 50), 
        accuracys, 
        color='black', 
        linestyle='dashed', 
        marker='o',
        markerfacecolor='blue', 
        markersize=10
    )
    plt.title('Accuracy at K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')

    plt.savefig("Accuracy of KNN.png")
    plt.clf()

    # Return the highest accuracy found
    print(f"Best accuracy for the KNN classifier was: {best_accuracy}")

"""
Name: classify_using_randomforest

Purpose: To classify our data through the usage of the random forest model

Input: data - Our data to train and test against

Output: None

Side-Effects: Produces a graph of the model's accuracy
"""
def classify_using_randomforest(data):

    n, d = data.shape

    train_num = int(n*.7)
    test_num = n-train_num

    train_data = data.iloc[:train_num,:]
    test_data = data.iloc[train_num:n,:]

    train_y_data = train_data[[TRUE_CLASS_COL]].to_numpy()
    train_x_data = train_data.drop(TRUE_CLASS_COL,axis = 1)

    test_y_data = test_data[[TRUE_CLASS_COL]].to_numpy().flatten()
    test_x_data = test_data.drop(TRUE_CLASS_COL,axis = 1)

    accuracys = []
    idxs = []
    best_accuracy = 0
    for estimators in range(10, 100):
        scaler = StandardScaler()
        scaler.fit(train_x_data)

        train_x_data = scaler.transform(train_x_data)
        test_x_data = scaler.transform(test_x_data)

        classifier = RandomForestClassifier(n_estimators=estimators)
        classifier.fit(train_x_data, train_y_data.ravel())

        y_pred = classifier.predict(test_x_data)
        if best_accuracy < accuracy_score(test_y_data, y_pred):
            best_accuracy = accuracy_score(test_y_data, y_pred)
        accuracys.append(accuracy_score(test_y_data, y_pred))
        idxs.append(estimators)

    plt.plot(
        idxs, 
        accuracys, 
        color='black', 
        linestyle='dashed', 
        marker='o',
        markerfacecolor='blue'
    )
    plt.title('RandomForest Accuracy measurements')
    plt.xlabel('Estimators')
    plt.ylabel('Accuracy')

    plt.savefig("RandomForest Estimator.png")
    plt.clf()

    print(f"Best accuracy for the Random Forest classifier was: {best_accuracy}")

"""
Name: classify_using_kmeans

Purpose: To classify our data using the KMeans model

Input: data - Our data to attempt to classify
       num_clusters - The number of clusters for our model

Output: None

Side-Effects: Reports the accuracy of the kmeans algorithm
"""
def classify_using_kmeans(data, num_clusters = 2):

    n, d = data.shape

    train_num = int(n*.7)
    test_num = n-train_num

    train_data = data.iloc[:train_num,:]
    test_data = data.iloc[train_num:n,:]

    train_y_data = train_data[[TRUE_CLASS_COL]].to_numpy()
    train_x_data = train_data.drop(TRUE_CLASS_COL,axis = 1)

    test_y_data = test_data[[TRUE_CLASS_COL]].to_numpy().flatten()
    test_x_data = test_data.drop(TRUE_CLASS_COL,axis = 1)

    scaler = StandardScaler()
    scaler.fit(train_x_data)

    train_x_data = scaler.transform(train_x_data)
    test_x_data = scaler.transform(test_x_data)

    classifier = KMeans(n_clusters = num_clusters)
    classifier.fit(train_x_data, train_y_data.ravel())

    y_pred = classifier.predict(test_x_data)
    accuracy = accuracy_score(test_y_data, y_pred)

    print(f"KMeans accuracy with determining desirable schools: {accuracy}")

"""
Name: classify_using_logistic_regression

Purpose: To classify our data using logistic regression

Input: data - Our data to classify

Output: None

Side-Effects: Reports the accuracy of the model
"""
def classify_using_logistic_regression(data):
    n, d = data.shape

    train_num = int(n*.7)
    test_num = n-train_num

    train_data = data.iloc[:train_num,:]
    test_data = data.iloc[train_num:n,:]

    train_y_data = train_data[[TRUE_CLASS_COL]].to_numpy()
    train_x_data = train_data.drop(TRUE_CLASS_COL,axis = 1)

    test_y_data = test_data[[TRUE_CLASS_COL]].to_numpy().flatten()
    test_x_data = test_data.drop(TRUE_CLASS_COL,axis = 1)

    scaler = StandardScaler()
    scaler.fit(train_x_data)

    train_x_data = scaler.transform(train_x_data)
    test_x_data = scaler.transform(test_x_data)

    classifier = LogisticRegression()
    classifier.fit(train_x_data, train_y_data.ravel())

    y_pred = classifier.predict(test_x_data)
    accuracy = accuracy_score(test_y_data, y_pred)

    print(f"Logistic Regression accuracy with determining desirable schools: {accuracy}")

if __name__=="__main__":
    main()