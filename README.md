# Disaster Response Pipelines
>  Utilize disaster data from Figure Eight to build a model for an API that classifies disaster messages.

## Table of Contents
1. [About this Project](#aboutthisproject)
    - [Built With](#builtwith)
2. [File Descriptions](#fileDes)
3. [Results](#results)
4. [Future Improvements](#improvements)
5. [Licensing, Authors, and Acknowledgements](#ack)
6. [Instructions](#instructions)

<a name='aboutthisproject'>

## About this project

This project uses thousands of real messages that were sent during disaster events. Each message is falling into several categories and there are 36 categories in total. Example of categories includes Food, Water, Money, Shelter, etc..

The main goal of this project is to build a model to categorize these events so that one can send the messages to an appropriate disaster relief agency. Also, it includes a web app as a final product where an emergency worker can input a new message and get classification results. 

<a name='builtwith'>
         
#### Built With
- Bootstrap
- JQuery
- Plotly 
- Flask

<a name='fileDes'>
    
## File Description

1. [data](https://github.com/Jiahuili858/Disaster-Response-Pipeline/tree/master/data) foler:
- disaster_catogories.csv: dataset contains all the categories
- disaster_messages.csv: dataset contains all the messages 
- process_data.py: script of ETL pipeline to process data
- DisasterResponse.db: output of the ETL pipeline, which stores the clean data (including both categories and messages) 

2. [models](https://github.com/Jiahuili858/Disaster-Response-Pipeline/tree/master/models) folder:
- train_classifier.py: script of Machine learning pipeline to predict classification for 36 categories
- classifier.pkl: output of the machine learning pipeline, which saves the model to a pickle file

3. [app](https://github.com/Jiahuili858/Disaster-Response-Pipeline/tree/master/app) folder
- run.py: flask file to run app
- templates includes two html files for the web app

4. Individual Output Performance.csv: saved the model performance (accuracy, precision, recall, f1-score) of each output category

<a name='results'>

## Results
- Average accuracy:  0.949773285872
- Average precision:  0.941253658577
- Average recall:  0.949773285872
- Average F1-Score:  0.938131834751

<a name='improvements'>
    
## Future Improvements

1. Tune more hyper-parameters in GridSearchCV to improve model performance
2. Yield a higher quality of dataset by including more works in processing data, such as removing rare words, correcting spelling
3. Use complex features, such as n-grams 

<a name='ack'>

## Licensing, Authors, and Acknowledgements
Data is from Figure Eight and the starter code is provided by Udacity.

<a name='instructions'>

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        <br />```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
    - To run ML pipeline that trains classifier and saves
        <br />```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```

2. Run the following command in the app's directory to run your web app.
    ```python run.py```

3. Go to http://0.0.0.0:3001/




