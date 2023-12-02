# Sagemaker-Housing-Prices
CPSC 454 Group project

Group Members:

Data Science/Analysis:
Niccolo Chuidian, Chloe Truong

Model training/ Machine Learning:
Adrian Charbonneau, Nayeli Umana


Requirements:

Create Amazon S3 bucket named 'housingdatacsv'

Import clean_house_data.csv into housingdata csv

Create folder 'module_data' in housingdatacsv main directory

Create jupyter notebook instance with kernal conda_python3

Import code from AIHousingPrices.py to new jupyter notebook isntance and run.

Model will take up to 10 minutes to train and deploy.

Clean up:

Open sagemaker console and under inference choose endpoints and delete all.

Under inference choose endpoint configurations, delete all.

Under inference choose models, delete all.

Under notebook, choose notebook isntances and choose stop, then delete.

Open Amazon S3 console and delete the bucket.

Open Amazon cloudwatch console and delete all of the log groups that contain /aws/sagemaker
