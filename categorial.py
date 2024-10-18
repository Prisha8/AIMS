import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.preprocessing import OneHotEncoder

user_behavior = pd.read_csv("/Users/tanyagulati/Downloads/user_behavior_dataset.csv")

y = user_behavior['User Behavior Class']
X = user_behavior.drop(['User ID', 'User Behavior Class'], axis =1, )

user_behavior = user_behavior.dropna()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state =0)

print(X_train.head())

def mae(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=50, random_state=0)
    
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    return mean_absolute_error(y_valid, pred)

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
good_label_cols = [col for col in object_cols if set(X_valid[col]).issubset(set(X_valid[col]))]
bad_label_cols = list(set(object_cols)- set(good_label_cols))

label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

ordinal_encoder = OrdinalEncoder()

label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols]= ordinal_encoder.transform(X_valid[good_label_cols])

print("mean absolute error from ordinal encoding is:")
print(mae(label_X_train, label_X_valid, y_train, y_valid))

OH_encoder = OneHotEncoder(handle_unknown= 'ignore', sparse_output= False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid))

OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

label2_X_train = X_train.drop(object_cols, axis=1)
label2_X_valid = X_valid.drop(object_cols, axis=1)

OH_X_train = pd.concat([OH_cols_train, label2_X_train], axis=1)
OH_X_valid = pd.concat([OH_cols_valid, label2_X_valid], axis=1)

OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("mean absolute error from OH encoding:")
print(mae(OH_X_train, OH_X_valid, y_train, y_valid))













