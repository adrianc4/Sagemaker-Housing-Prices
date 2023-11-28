import boto3
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

# Assuming boto3 is already configured with AWS credentials

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