#import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath, table_name='disaster'):
    """ load data from database

    Args:
        database_filepath: file path of the database
        table_name: name of database table 

    Returns:
        X: preditors
        Y: target variables
        cateogry_names: list of the category names

    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """ Tokenization to process text data

    Args:
        text: raw text data to be used for tokenization

    Returns:
        clean_tokens: cleaned tokens 

    """

    remove_list = [
        r'https?\s?:\s?//[A-Za-z0-9./]+', #URL links
        r'@[A-Za-z0-9]+', #@mentions
        r'#[A-Za-z0-9]+', #hashtags
        r"[^a-zA-Z0-9]", #punctuation
        r'\d+' #numbers
    ]

    #remove noisy information
    for i in remove_list:
        text = re.sub(i, ' ', text)
        
    #tokenize
    tokens = word_tokenize(text)
    
    #lemmatize, lowercase, remove whitespace
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() 
        clean_tokens.append(clean_tok)
        
    #remove stopwords
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")] 
    
    return clean_tokens
    

def build_model():
    """ Using pipeline with GridSearch 

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state = 123),n_jobs=-1))
    ])

    parameters = {
       'clf__estimator__n_estimators': [10,50,100],
        'clf__estimator__min_samples_leaf': [1,2,4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_macro')

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ evalute the model using accuracy, precision, recall and f1-score

    """ 
    # prediction
    Y_pred = model.predict(X_test)
    
    # print out best parameters of the model
    print("Best paramters: ", model.best_params_ )
    
    
    i = 0
    acc, pre, recall, f1 = 0, 0, 0, 0
    ind_perf = {'Category':[], 'Accuracy': [], 'Precision': [], 'Recall':[], 'f1-Score':[]}
    
    for col in category_names:
        
        # save individual output performance
        ind_perf['Category'].append(col)
        ind_perf['Accuracy'].append(accuracy_score(Y_test[col], Y_pred[:, i]))
        ind_perf['Precision'].append(precision_score(Y_test[col], Y_pred[:, i], average='weighted'))
        ind_perf['Recall'].append(recall_score(Y_test[col], Y_pred[:, i], average='weighted'))
        ind_perf['f1-Score'].append(f1_score(Y_test[col], Y_pred[:, i], average='weighted'))
        
        # calculate overall output performance 
        acc += accuracy_score(Y_test[col], Y_pred[:, i])
        pre += precision_score(Y_test[col], Y_pred[:, i], average='weighted')
        recall += recall_score(Y_test[col], Y_pred[:, i], average='weighted')
        f1 += f1_score(Y_test[col], Y_pred[:, i], average='weighted')
        i = i + 1

    print("\nOverall Performance.....")
    print("Average accuracy: ", acc/len(category_names))
    print("Average precision: ", pre/len(category_names))
    print("Average recall: ", recall/len(category_names))
    print("Average F1-Score: ", f1/len(category_names))
    
    performance = pd.DataFrame.from_dict(ind_perf) 
    performance.to_csv('Individual Output Performance.csv')


def save_model(model, model_filepath):
    """ Save model results as pickle file

        Args:
            model_name: name of the pick file to be saved
            model: model object returned from function build_model() 
        
        Returns: 
            None
    """
    with open('{}'.format(model_filepath), 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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