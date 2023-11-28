# Imports for data manipulation
import boto3
import pandas as pd
import os
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Imports for model training
import sagemaker
from sagemaker.debugger import Rule, ProfilerRule, rule_configs
from sagemaker.session import TrainingInput
from IPython.display import FileLink, display
# Imports for model deployment
from sagemaker.serializers import CSVSerializer
# Imports for model evaluation
import numpy as np
import io
import sklearn.metrics as metrics

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

# Converting categorical variable ocean_proximity into one-hot encoding
df_encoded = pd.get_dummies(df, columns=['ocean_proximity'])

# Split data into features (X) and target (y)
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

# Splitting training set into validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1)

# Alligning dataset by MedianHouseValue
train = pd.concat([pd.Series(y_train, index=X_train.index,
                             name='MedianHouseValue'), X_train], axis=1)
validation = pd.concat([pd.Series(y_val, index=X_val.index,
                                  name='MedianHouseValue'), X_val], axis=1)
test = pd.concat([pd.Series(y_test, index=X_test.index,
                            name='MedianHouseValue'), X_test], axis=1)

# Converting train and validation dataframe objects to CSV
train.to_csv('train.csv', index=False, header=False)
validation.to_csv('validation.csv', index=False, header=False)

# Location CSV files are saved, prefix is the folder name
bucket = 'housingdatacsv'
prefix = 'model_data'

boto3.Session().resource('s3').Bucket(bucket).Object(
    os.path.join(prefix, 'data/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(
    os.path.join(prefix, 'data/validation.csv')).upload_file('validation.csv')

# Model Deployment -----------------------------------------------------

# Retrieve information from current sagemaker session and save into variable
region = sagemaker.Session().boto_region_name
print("AWS Region: {}".format(region))

role = sagemaker.get_execution_role()
print("RoleArn: {}".format(role))

s3_output_location = f's3://{bucket}/{prefix}/xgboost_model'

container = sagemaker.image_uris.retrieve("xgboost", region, "1.2-1")
print(container)

# Creating XGBoost estiamtor
xgb_model = sagemaker.estimator.Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    volume_size=5,
    output_path=s3_output_location,
    sagemaker_session=sagemaker.Session(),
    rules=[
        Rule.sagemaker(rule_configs.create_xgboost_report()),
        ProfilerRule.sagemaker(rule_configs.ProfilerReport())
    ]
)

# Setting hyperparameters for estimator
xgb_model.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.7,
    # Regression subjective method to predict housing prices
    objective="reg:squarederror",
    num_round=1000
)

# Using traininginput class to configure data input flow for training
train_input = TrainingInput(
    "s3://{}/{}/{}".format(bucket, prefix, "data/train.csv"),
    content_type="csv"
)
validation_input = TrainingInput(
    "s3://{}/{}/{}".format(bucket, prefix, "data/validation.csv"),
    content_type="csv"
)

# Starting model training by calling estimator's fit method
# wait = true displays progress logs and waits until trainign is complete
xgb_model.fit({"train": train_input, "validation": validation_input},
              wait=True)

# Specifying S3 bucket URI to store debugger training reports
rule_output_path = (xgb_model.output_path + "/"
                    + xgb_model.latest_training_job.job_name
                    + "/rule-output")


# IPython script to get file link of XGBoost training report
display("Click link below to view the XGBoost Training report",
        FileLink("CreateXgboostReport/xgboost_report.html"))


# Return file link of debugger of instance resource utilization
profiler_report_name = [rule["RuleConfigurationName"]
                        for rule in (xgb_model.latest_training_job.
                        rule_job_summary())
                        if "Profiler" in rule["RuleConfigurationName"]][0]

display("Click link below to view the profiler report", FileLink(
    profiler_report_name+"/profiler-output/profiler-report.html"))

xgb_model.model_data

# Model Deployment ---------------------------------------

# Hosting model through EC2 using deploy method from xgb_model estimator
xgb_predictor = xgb_model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    serializer=CSVSerializer()
)

# Retrieve endpoint name
# Endpoint stays active in ML instance to make,
# instantaneous predictions until shut down
xgb_predictor.endpoint_name

# Model Evaluation ------------------------------------------


# Function predicts each line of test set
def predict(data, rows=1000):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        buffer = io.StringIO()
        np.savetxt(buffer, array, delimiter=',', fmt='%g')
        predictions_csv = buffer.getvalue()
        predictions = ','.join([
            predictions, xgb_predictor.predict(predictions_csv).decode('utf-8')
        ])
    return np.fromstring(predictions[1:], sep=',')


# plotting predictions of test dataset on histogram
predictions = predict(test.to_numpy()[:, 1:])
plt.hist(predictions, bins=30, color='blue')
plt.title("Distribution of Predicted Values")
plt.xlabel("Predicted Value")
plt.ylabel("Frequency")
plt.show()

# Evaluation Metrics for Regression
# Assuming test.iloc[:, 0] is the true target values
true_values = test.iloc[:, 0]
print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(true_values,
                                                                predictions))
print("Mean Squared Error (MSE):", metrics.mean_squared_error(true_values,
                                                              predictions))
print("R-squared (Coefficient of Determination):",
      metrics.r2_score(true_values, predictions))

# Residuals Plot
residuals = true_values - predictions
# Purple color for residuals
plt.scatter(predictions, residuals, color='purple')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

# Predicted vs Actual Values Plot
# Blue for predicted
plt.scatter(predictions, true_values, color='blue', label='Predicted Values')
# Green for actual
plt.scatter(true_values, true_values, color='green', alpha=0.5, label='Actual Values')
plt.title("Predicted vs Actual Values")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.plot([true_values.min(), true_values.max()], [
    true_values.min(), true_values.max()], 'k--', lw=2)
plt.legend()
plt.show()