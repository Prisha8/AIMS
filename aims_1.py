import pandas as pd 
import numpy as np

user_behavior = pd.read_csv("/Users/tanyagulati/Downloads/user_behavior_dataset.csv")
print(user_behavior.head())

object_cols = [col for col in user_behavior.columns if user_behavior[col].dtype == 'object']
print(f"the categorial columns in the dataset are:  {object_cols}")

print(f"the unique values in Device Model are: { user_behavior['Device Model'].unique()}")
print(f"the unique values in Operating system are: { user_behavior['Operating System'].unique()}")
print(f"the unique values in gender are: {user_behavior['Gender'].unique()}")



map_DM = {'Google Pixel 5':0, 'OnePlus 9':1, 'Xiaomi Mi 11':2, 'iPhone 12':3, 'Samsung Galaxy S21':4}
map_gen = {'Male':0, 'Female':1}
map_OS = {'Android':0, 'iOS':1}

user_behavior['en_col_DM'] = user_behavior['Device Model'].map(map_DM)
user_behavior['en_col_gen'] = user_behavior['Gender'].map(map_gen)
user_behavior['en_col_OS'] = user_behavior['Operating System'].map(map_OS)

print("The dataset using ordinal encoding is:- ")
print(user_behavior.head())


for value in user_behavior['Device Model'].unique():
    user_behavior[value] = (user_behavior['Device Model'] == value).astype(int)

for gen_val in user_behavior['Gender'].unique():
    user_behavior[gen_val] = (user_behavior['Gender'] == gen_val).astype(int)

for os_val in user_behavior['Operating System'].unique():
    user_behavior[os_val] = (user_behavior['Operating System'] == os_val).astype(int)    

print("the dataset using one hot encoder is :-")
print(user_behavior.head())



     




