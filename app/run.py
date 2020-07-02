import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


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

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('disaster', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    message_counts = df.iloc[:,4:].sum().sort_values(ascending=False)
    category_names = list(message_counts.index)
    
    category_counts = df.iloc[:,4:].sum(axis=1).value_counts().sort_index(ascending=False)
    num_of_category = list(category_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=message_counts
                )
            ],

            'layout': {
                'title': 'Number of Message Per Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle':-30
                    
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=num_of_category,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count of Message"
                },
                'xaxis': {
                    'title': "Number of Categories a Message Belongs to"
                    
                }
            }
        }     
       
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()