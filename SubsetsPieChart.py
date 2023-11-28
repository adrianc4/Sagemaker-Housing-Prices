import boto3
import pandas as pd
import os
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Specify your bucket name and object key (file path in S3)
bucket_name = 'cl-sagemaker-cleanhousedata'
file_key = 'clean_house_data.csv'

# Create a client
s3_client = boto3.client('s3')

# Get the object from S3
obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
data = obj['Body'].read().decode('utf-8')

# Convert the string data to a pandas DataFrame
df = pd.read_csv(StringIO(data))

# Converting categorical variable ocean_proximity into one-hot encoding
df_encoded = pd.get_dummies(df, columns=['ocean_proximity'])

# Split data into features (X) and target (y)
X = df_encoded.drop('median_house_value', axis=1)
y = df_encoded['median_house_value']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.2, random_state=1)

# Splitting training set into validation set
X_train, X_val, y_train, y_val = train_test_split(
                                    X_train, y_train, test_size=0.25, random_state=1)

# Pie chart representing the proportions of training, validation, and test data 
train_size = len(X_train)
validation_size = len(X_val)
test_size = len(X_test)

# Data to plot
sizes = [train_size, validation_size, test_size]
labels = ['Training', 'Validation', 'Test']
colors = ['lightcoral', 'lightskyblue', 'lightgreen']
explode = (0.1, 0, 0)  # explode 1st slice

# Plot
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')

# Display the pie chart
plt.show()
