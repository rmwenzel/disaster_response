import sys
import pandas as pd
import pickle as pkl
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,
                             precision_recall_fscore_support,
                             fbeta_score, make_scorer)
from sklearn.utils import parallel_backend
from sklearn.multioutput import MultiOutputClassifier

nltk.download(['punkt', 'stopwords', 'wordnet'])

import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    """Load feature and response data from SQL database file."""
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filepath.replace('.db', '').split('/')[-1]
    df = pd.read_sql_table(table_name, engine)
    # message text as features
    X = df['message']
    # message category as response
    Y = df[df.columns.drop(['id', 'message', 'genre'])]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """Tokenize messages."""
    # replace urls with placeholder
    url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # convert to lower case
    text = text.lower()
    # remove punctuation characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # word tokens
    words = word_tokenize(text)
    # remove words with digits and stop words
    words = [w for w in words if w not in stopwords.words('english')]
    # lemmatize
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed


def build_model():
    """Build gridseach cv ML pipeline."""
    # pipeline
    rf_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, min_df=2)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    # grid search parameters
    params = {'clf__estimator__class_weight': ['balanced'],
              'clf__estimator__max_depth': [3, 5],
              'clf__estimator__max_features': ['auto'],
              'clf__estimator__min_samples_leaf': [2],
              'clf__estimator__n_estimators': [20],
              'clf__estimator__n_jobs': [-1],
              'clf__estimator__random_state': [27],
              'clf__n_jobs': [-1]
              }
    # use fbeta scorer, favoring recall over precision, and taking
    # micro average to reflect class imbalances
    fbeta_scorer = make_scorer(fbeta_score, beta=3, average='micro')
    # small grid search for tuning
    rf_pipeline_cv = GridSearchCV(rf_pipeline,
                                  params,
                                  scoring=fbeta_scorer,
                                  n_jobs=-1,
                                  cv=5,
                                  pre_dispatch='2*n_jobs',
                                  verbose=1)
    return rf_pipeline_cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Print classification metrics per category and overall."""
    Y_pred = model.predict(X_test)
    # print accuracy, precision and recall for each category
    for (i, category) in enumerate(category_names):
        print(f'For message category {category}:\n')
        print(classification_report(Y_pred[:, i], Y_test[category]))

    prec, rec, f1, _ = precision_recall_fscore_support(Y_test, Y_pred,
                                                       average='macro')
    fbeta = fbeta_score(Y_test, Y_pred, beta=3, average='macro')
    print()
    print(f'Macro average precision: {prec}')
    print(f'Macro average recall: {rec}')
    print(f'Macro average f1: {f1}')
    print(f'Macro average fbeta, beta=3: {fbeta}')

    prec, rec, f1, _ = precision_recall_fscore_support(Y_test, Y_pred,
                                                       average='micro')
    fbeta = fbeta_score(Y_test, Y_pred, beta=3, average='micro')
    print()
    print(f'Micro average precision: {prec}')
    print(f'Micro average recall: {rec}')
    print(f'Micro average f1 score: {f1}')
    print(f'Micro average fbeta, beta=3: {fbeta}')


def save_model(model, model_filepath):
    """Pickle model to filepath."""
    with open(model_filepath, 'wb') as model_file:
        pkl.dump(model, model_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2)

        with parallel_backend('multiprocessing'):
            print('Building model...')
            model = build_model()

            print('Training model...')
            model.fit(X_train, Y_train)
            # get best estimator from grid search
            model = model.best_estimator_

            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, category_names)

            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)

            print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
