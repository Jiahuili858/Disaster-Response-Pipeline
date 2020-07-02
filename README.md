# Disaster Response Pipelines
>  Utilize disaster data from Figure Eight to build a model for an API that classifies disaster messages.

## Table of Contents
1. [About this Project](#aboutthisproject)
    - [Built With](#builtwith)
2. [File Descriptions](#fileDes)
3. [Results](#results)
4. [Future Improvements](#improvements)
5. [Instructions](#instructions)
6. [Outlook of the Flask Web App](#app)
7. [Licensing, Authors, and Acknowledgements](#ack)

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

1. [data](https://github.com/Jiahuili858/Disaster-Response-Pipeline/tree/master/data) folder:
	- disaster_catogories.csv: dataset contains all the categories
	- disaster_messages.csv: dataset contains all the messages 
	- process_data.py: script of ETL pipeline to process data
	- DisasterResponse.db: output of the ETL pipeline, which stores the clean data (including both categories and messages) 

2. [models](https://github.com/Jiahuili858/Disaster-Response-Pipeline/tree/master/models) folder:
	- train_classifier.py: script of Machine learning pipeline to predict classification for 36 categories
	- classifier.pkl: output of the machine learning pipeline, which saves the model to a pickle file

3. [app](https://github.com/Jiahuili858/Disaster-Response-Pipeline/tree/master/app) folder:
	- run.py: flask file to run app
	- templates includes two html files for the web app

4. [Individual Output Performance.csv](https://github.com/Jiahuili858/Disaster-Response-Pipeline/blob/master/Individual%20Output%20Performance.csv): saved the model performance (accuracy, precision, recall, f1-score) of each output category

<a name='results'>

## Results
Here is the overall model performance:
- Average accuracy:  0.949773285872
- Average precision:  0.941253658577
- Average recall:  0.949773285872
- Average F1-Score:  0.938131834751

 ***notes:** As this dataset is imbalanced, weighted average is taking when calculate precision, reall and f1-score for each output category.*

<a name='improvements'>
    
## Future Improvements

- Tune more hyper-parameters in GridSearchCV to improve model performance
- Yield a higher quality of dataset by including more works in processing data, such as removing rare words, correcting spelling
- Use complex features, such as n-grams 

<a name='instructions'>

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        <br />```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
    - To run ML pipeline that trains classifier and saves
        <br />```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```

2. Run the following command in the project's root directory to run your web app.
    ```python app/run.py```

3. Go to http://0.0.0.0:3001/

<a name='app'>
    
## Outlook of the Flask Web App

![alt text](https://github.com/Jiahuili858/Disaster-Response-Pipeline/blob/master/images/Screen%20Shot%201.png)

![alt text](https://github.com/Jiahuili858/Disaster-Response-Pipeline/blob/master/images/Screen%20Shot%202.png)

<a name='ack'>

## Licensing, Authors, and Acknowledgements
Data is from Figure Eight and the starter code is provided by Udacity.





