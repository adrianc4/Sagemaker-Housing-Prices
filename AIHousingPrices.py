import boto3
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Specify your bucket name and object key (file path in S3)
bucket_name = 'housingdatacsv'
file_key = 'clean_house_data.csv'

# Create a client
s3_client = boto3.client('s3')

# Get the object from S3
obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
data = obj['Body'].read().decode('utf-8')

# Convert the string data to a pandas DataFrame
df = pd.read_csv(StringIO(data))

df_encoded = pd.get_dummies(df, columns=['ocean_proximity'])

X = df_encoded.drop('median_house_value', axis=1)
y = df_encoded['median_house_value']

feature_names = list(X.columns)
feature_names

print(X.describe())
hist = X.hist(bins=30, sharey=True, figsize=(20, 10))
plt.show()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)

X_display = df

# Aligning X_display with X_train and X_test
X_train_display = X_display.loc[X_train.index]
X_val_display = X_display.loc[X_test.index]

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1)

train = pd.concat([pd.Series(y_train, index=X_train.index,
                             name='MedianHouseValue'), X_train], axis=1)
validation = pd.concat([pd.Series(y_val, index=X_val.index,
                                  name='MedianHouseValue'), X_val], axis=1)
test = pd.concat([pd.Series(y_test, index=X_test.index,
                            name='MedianHouseValue'), X_test], axis=1)

train

validation

test

# Converting train and validation dataframe objects to CSV
train.to_csv('train.csv', index=False, header=False)
validation.to_csv('validation.csv', index=False, header=False)
