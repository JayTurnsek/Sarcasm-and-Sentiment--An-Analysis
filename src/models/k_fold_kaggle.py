# Title: k_fold_kaggle.py
# Author: Jay Turnsek
# Date Modified: 2023-04-03
# 
# Used to validate the polarity lexicon, logistic regression, and SA-LR
# models on the Kaggle dataset using k-fold validation with k = 5.
# Outputs a JSON file holding the precision, recall, and f1 score 
# metrics for each model at each k fold.
# call as 'python k_fold_kaggle.py output_filename.json' in console, 
# with directory structure maintained.

import pandas as pd
from nltk.tokenize import TweetTokenizer
import sys
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score
import json

## Polarity lexicon classifier implementation. Used to classify a string of
## text based on sarcasm or not.
class lexicon:
    def __init__(self, s_words, ns_words):
        self.tokenizer = TweetTokenizer()
        self.train(s_words, ns_words)

    def train(self, s_words, ns_words):

        # Populate the sarcastic and non sarastic word sets for use in
        # classification
        self.s_words = s_words
        self.ns_words = ns_words

    def classify(self, test_instance):

        # Count instances of sarcastic and non-sarcastic words.
        classes = {
            "sarcastic": 0,
            "nonsarcastic": 0
        }
        for word in self.tokenizer.tokenize(test_instance):
            if word in self.s_words:
                classes["sarcastic"] += 1
            if word in self.ns_words:
                classes["nonsarcastic"] += 1
        
        # Return the class with the higher number of words pertaining to class.
        if classes["sarcastic"] > classes["nonsarcastic"]:
            return 1
        elif classes["sarcastic"] < classes["nonsarcastic"]:
            return 0

        # In the event of a tie, flip a coin!
        else:
            return random.randint(0, 1)

# Logistic regression baseline
class LR:
    def __init__(self, train_tweets, train_labels):

        # Initialize model with the Logistic Regression Model provided
        # by SkLearn. n_jobs=-1 chosen to utilize all CPU cores
        self.model = LogisticRegression(n_jobs=-1)
        self.train(train_tweets, train_labels)

    def train(self, train_tweets, train_labels):

        # train_tweets must be an array of tweets with SOME numerical
        # encoding. In this example we used tf-idf.
        
        # Call fit function of sklearn
        self.model.fit(train_tweets, train_labels)


    def classify(self, test_instance):

        # Test instance must be an instance of numerical encoding of tweet,
        # consistent with training data.

        # Call predict function of sklearn model. returns label
        return self.model.predict(test_instance)

# SENTIMENT-AWARE logistic regression model, build upon previous LR model.
class LRSent:
    def __init__(self, train_tweets, train_labels, sentiment):

        # Initialize THREE models for each sentiment label.
        self.pos_model = LogisticRegression(n_jobs=-1)
        self.neutral_model = LogisticRegression(n_jobs=-1)
        self.neg_model = LogisticRegression(n_jobs=-1)
        self.train(train_tweets, train_labels, sentiment)

    def train(self, train_tweets, train_labels, sentiment):

        # Split dataset into bins that hold each sentiment's tweets.
        split_train_tweets = {}
        split_train_labels = {}
        for sentiment_label in ['positive', 'negative', 'neutral']:
            sentiment_i = np.where(sentiment == sentiment_label)[0]
            sentiment_partition_tweets = train_tweets[sentiment_i]
            sentiment_partition_labels = train_labels[sentiment_i]
            split_train_tweets[sentiment_label] = sentiment_partition_tweets
            split_train_labels[sentiment_label] = sentiment_partition_labels

        # Fit each model using the corrosponding sentiment dataset.
        self.pos_model.fit(split_train_tweets['positive'], split_train_labels['positive'])
        self.neutral_model.fit(split_train_tweets['neutral'], split_train_labels['neutral'])
        self.neg_model.fit(split_train_tweets['negative'], split_train_labels['negative'])

    def classify(self, test_instance, instance_sentiment):

        # First check which sentiment instance is classified as,
        # then send to corrosponding LR model for classification.
        if instance_sentiment == 'positive':
            return self.pos_model.predict(test_instance)
        elif instance_sentiment == 'neutral':
            return self.neutral_model.predict(test_instance)
        elif instance_sentiment == 'negative':
            return self.neg_model.predict(test_instance)
        else:
            raise Exception("Invalid sentiment in test instance. Sentiment should be in ['positive', 'neutral', 'negative'].")



if __name__ == "__main__":
    # called using python k_fold_kaggle.py dataset_path output_path sentiment_path
    # get command line args
    outputfile = sys.argv[1]

    # prep dataset first, then apply kfold validation
    dataset = pd.read_csv("../../data/kaggleTweets/dataset.csv", encoding='utf-8')
    tweets, labels = dataset['tweets'].tolist(), dataset['class'].to_numpy()

    # initialize metrics data dictionary
    metrics = {}
    class_labels = [0, 1]
    for method in ['lexicon', 'lr', 'lrsent']:
        
        # initialize the metrics for this model.
        metrics[method] = {
            'recall': [],
            'precision': [],
            'f1 score': []
        }

        # Only initialize sentiment data on lrsent model
        if method == "lrsent":
            s_file = open("../../data/kaggleTweets/sentiment.txt", encoding='utf-8')
            sentiment = [x.split()[0] for x in s_file.readlines()]

        # K-Fold validation with k = 5
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        for train_i, test_i in skf.split(np.zeros(len(tweets)), labels):
            
            # Partition data with k fold indexes
            train_tweets_fold = [tweets[i] for i in train_i]
            test_tweets_fold = [tweets[i] for i in test_i]
            train_labels_fold, test_labels_fold = labels[train_i], labels[test_i]
            
            if method == "lexicon":

                # Get sarcastic and nonsarcastic words to populate polarity lexicon
                # Use tweet tokenizer to split tweets into word tokens
                s_words, ns_words = [], []
                tokenizer = TweetTokenizer(preserve_case=False)
                for tweet, label in zip(train_tweets_fold, train_labels_fold):

                    # stop words!
                    curTweet = [x for x in tokenizer.tokenize(tweet) if not x in ["#sarcasm", "#sarcastic", "sarcastic", "sarcasm"]]

                    # Add to corrosponding array
                    if label == 1:
                        s_words.extend(curTweet)
                    elif label == 0:
                        ns_words.extend(curTweet)
                    else:
                        raise Exception("Incorrect label.")
                
                # convert to set.
                s_words, ns_words = set(s_words), set(ns_words)

                # initialize model with populated lexicon
                model = lexicon(s_words, ns_words)

            elif method == "lr":
                
                # transform textual data to numeric representation, in our case tf-idf.
                vectorizer = TfidfVectorizer(stop_words=["#sarcasm", "#sarcastic", "sarcastic", "sarcasm"], max_features=20)     # Why isnt this working right rn
                vectorizer.fit(train_tweets_fold)
                train_tweets_fold = vectorizer.transform(train_tweets_fold)
                test_tweets_fold = vectorizer.transform(test_tweets_fold)

                # Initialize model with encoded dataset
                model = LR(train_tweets_fold, train_labels_fold)
            
            
            elif method == "lrsent":

                # transform textual data to numeric representation, in our case tf-idf.
                vectorizer = TfidfVectorizer(stop_words=["#sarcasm", "#sarcastic", "sarcastic", "sarcasm"], max_features=20)     # Why isnt this working right rn
                vectorizer.fit(train_tweets_fold)
                train_tweets_fold = vectorizer.transform(train_tweets_fold)
                test_tweets_fold = vectorizer.transform(test_tweets_fold)
                train_sentiment = np.array([sentiment[i] for i in train_i])
                test_sentiment = np.array([sentiment[i] for i in test_i])

                # Initialize model with encoded dataset
                model = LRSent(train_tweets_fold, train_labels_fold, train_sentiment)

            else:
                raise Exception("Invalid model name. Pick from 'lexicon', 'lr', or 'lrsent'. ")

            # complete prediction
            if method == "lrsent":
                predictions = np.array([model.classify(x, x_sent) for x, x_sent in zip(test_tweets_fold, test_sentiment)])
            else:
                predictions = np.array([model.classify(x) for x in test_tweets_fold])
            
            # add corrosponding metrics.
            metrics[method]['recall'].append(recall_score(test_labels_fold, predictions, labels=class_labels))
            metrics[method]['precision'].append(precision_score(test_labels_fold, predictions, labels=class_labels))
            metrics[method]['f1 score'].append(f1_score(test_labels_fold, predictions, labels=class_labels))


    # Convert dictionary to JSON string
    metrics_str = json.dumps(metrics)

    # Write JSON string to file
    with open(outputfile, 'w') as f:
        f.write(metrics_str)