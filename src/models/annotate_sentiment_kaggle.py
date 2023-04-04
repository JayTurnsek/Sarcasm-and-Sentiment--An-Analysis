# Title: annotate_sentiment_kaggle.py
# Author: Jay Turnsek
# Date Modified: 2023-04-03
# 
# Used to annotate the tweets from the tweetEval dataset with a sentiment label.
# Uses relative paths, takes no command line args
import pandas as pd
import tweetnlp



if __name__ == "__main__":
    # Open dataset from Kaggle
    dataset = pd.read_csv("../../data/kaggleTweets/dataset.csv", encoding='utf-8')
    tweets, labels = dataset['tweets'].tolist(), dataset['class'].to_numpy()


    # add sentiment labelling, write to new file.
    model = tweetnlp.load_model('sentiment')
    outf = open("../../data/kaggleTweets/sentiment.txt", "w")
    for i in range(len(tweets)):
        predict = model.sentiment(tweets[i])
        outf.write(predict['label'] + "\n")

    outf.close()

