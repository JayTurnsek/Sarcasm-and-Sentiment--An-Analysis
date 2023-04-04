# Sarcasm and Sentiment, an Analysis
## Author: Jay Turnsek
## Date: 2023-04-03
# Summary: 
This is a set of files used to complete the tests outlined in the report 'Sarcasm and Sentiment, an Analysis'.
The datasets are contained in the /data directory, and runnable scripts in src/models.
Sentiment annotation is already complete if you just want to run the testing scripts 'k_fold_kaggle.py' and 'test_tweeteval.py',
though the original annotation scripts can also be ran if wanted. The folder structure must be maintained to have scripts run correctly, as relative pathing was used in the opening/writing of files; with the exception of result output JSONs.
#
# How to run:
To try the scripts for yourself, just copy the repository and unzip into your directory of choice, then open a terminal in src/models. There are 4 runnable scripts:
1. The k-fold validation of the three NLP models on the larger Kaggle dataset, which outputs a JSON of the performance metrics of each model (precision, accuracy, F1-score). This is ran as such:
```
python k_fold_kaggle.py output_filename.json
```
2. The 75/25 testing of the three NLP models on the smaller TweetEval dataset, which outputs a JSON of the performance metrics of each model (precision, accuracy, F1-score). This is ran as such:
```
python test_tweeteval.py output_filename.json
```
3. The sentiment annotating of the kaggle dataset, called without command line args:
```
python annotate_sentiment_kaggle.py
```
4. The sentiment annotating of the kaggle dataset, called without command line args:
```
python annotate_sentiment_tweeteval.py
```
Plots and tables were generated from the outputted JSONs when shown in the report, these generative scripts were omitted as they were outside the scope of the criteria for the CSCI361 course project.

# Awknowledgements:
Thank you Dr. Milton King for helping me understand NLP, and making the class interesting and motivating throughout!  
Cheers,  
Jay  