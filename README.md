# Disaster Response NLP Pipeline and Web App

A simple Flask web app that uses NLP to classify messages into categories related to disaster reponse.

The user can input a text message and see which categories it is classified into, as well as view visualizations of the dataset used for training, [Multilingual Disaster Response Messages](https://appen.com/datasets/combined-disaster-response-data/).

The final model is optimized for [F-beta score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html), which is similar to f1 score but allows to one weight recall higher than precision (or vice versa). Emphasizing recall over precision makes more sense for this application.

For more discussion, and more detailed analysis of the data and the ETL and ML pipelines, see the notebooks.

### Instructions:

1. Run the following commands in the project's root directory to set up the database and model.

    - To run the ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the `app` directory to run the web app.
    `python run.py`

3. Go to `http://0.0.0.0:3001/` to use the web app


### Structure:

```
.
├── README.md
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── DisasterResponse.db
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py
├── models
│   └── train_classifier.py
└── notebooks
    ├── ETL.ipynb
    └── ML.ipynb
```

Directory overview:
- `app` - html templates and script for running the Flask app.
- `data` - original data as csvs, SQL database file for cleaned and processed data, and script for doing the processing.
- `model` - classifier model training script.
- `notebooks` - Jupyter notebooks for data processing, exploration, and machine learning model building and evaluation. Most code from these notebooks is used to create the data processing and training scripts.

