# IEEE-Latin-America
Code to test all variables in NARX models. In this case we use a vector of nine variables but the script works with vector of n variables
Instructions:
This Github container provides the elements to develop NARX models with all variable combinations. The All Variable Combination Algorithm (AVCA) works with a matrix of n rows and m columns where columns are the variables and rows are the measurements in the time t.
Data for the test and results is in folder 01 Data/CleanData. In folder 02 Code, we have three folders:
## 01 Script CC. 
The results for NARX models are provided using input vectors obtained with the Collinearity and Causality test proposed in “**Input vector selection in NARX models using statistical techniques to improve the generated power forecasting in PV systems**” IEEE Latin America Transactions Journal. Here you will find four MATLAB scripts.
## 02 Script AVCA. 
The ten best NARX models obtained with the AVCA and their respective input vectors are provided. I have provided a brief theoretical framework and reference in the previously mentioned paper. Here you will find ten MATLAB scripts.
## 03 Script All Combination. 
Provides the MATLAB script to figure out the best ten input vectors testing all possible variable combinations.
Once you load the dataset to MATLAB, you can run the scripts.
