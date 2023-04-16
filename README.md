# LoanDefault
This repo containts an ML project to predict defaults of Lending Club loans. It is deployable via a simple Flask webapp on Google Cloud Platform (GCP). 
An XGBoost model here is a simplified version of my model from [Kaggle](https://www.kaggle.com/code/pmykola/lending-club-loan-default-prediction/edit/run/109869105). It is built in 'loans_modeling_' notebooks.
This project is deployable via GCP App Engine. Free GCP trial subscription is sufficient to deploy the model.

<br>
<br>

**Steps to build and deploy the model on GCP:**
1. Clone this repo.
2. Cd into loans-app.
3. 
```
gcloud init
gcloud app deploy
```
4. Follow the link, shown in the console after the previous command to verify that webapp works.




Notes:
- To use target encoder at prediction time, I can use this object directly. To make it work, see this: https://stackoverflow.com/questions/57888291/how-to-properly-pickle-sklearn-pipeline-when-using-custom-transformer.

