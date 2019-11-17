import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler

# state column names 
col_names = ['age', 'workclass', 
'fnlwgt', 'education', 'education_num', 
'marital_status', 'occupation', 'relationship', 'race', 'sex',
'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'the label']

test_raw_nan = pd.read_csv('census-income.test.csv', names = col_names)

#replace the question marks in file with null values, making sure there is a space before the question mark to determine the rows 
test_raw_nan.replace(' ?', np.nan, inplace = True)

#drop missing values 
test_raw = test_raw_nan.dropna()

#For the mssing values 
print('test data contains {} rows with missing values'.format(len(test_raw_nan) - len(test_raw)))

# get_dummies to convert categorical variables into dummy variables
#iloc to retrieve rows from the dataframe
target_nans = pd.get_dummies(test_raw_nan).iloc[:,-1]
target_nans.value_counts()

print('nagative instances accounted for {}% of our times'.format(1221/(12435+3846)* 100))

test_raw

#creates a target column from the first column to the one before the last column 
test_target = pd.get_dummies(test_raw).iloc[:, -1]

#lists the target columns for the dummy variables 
test_target.head()

#returns the count of the unbalanced data
test_target.value_counts()
