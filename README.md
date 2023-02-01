# LoanDefault
This repo containts a ML project to predict defaults of Lending Club loans. It is deployable via a simple Flask webapp on Google Cloud Platform (GCP). 
The XGBoost model here is a simplified version of my model from Kaggle. It is built in 'loans_modeling_' notebooks.
This project is deployable via GCP App Engine. Free GCP trial is sufficient to deploy the model.

Steps to build and deploy the model on GCP:
1. Clone this repo.
2. Cd into loans-app.
3. 
```
gcloud init
gcloud app deploy
```
4. follow the link, shown in the console after the previous command to verify that webapp works.
