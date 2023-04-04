# Title: annotate_sentiment_tweeteval.py
# Author: Jay Turnsek
# Date Modified: 2023-04-03
#
# Used to annotate the tweets from the Kaggle dataset with a sentiment label.
# Uses relative paths, takes no command line args
import tweetnlp


if __name__ == "__main__":
    # load dataset
    train_tweets = open("../../data/tweeteval/train_tweets.txt", encoding="utf8")
    train_tweets = train_tweets.readlines()
    test_tweets = open("../../data/tweeteval/val_tweets.txt", encoding="utf8")
    test_tweets = test_tweets.readlines()


    # load model
    model = tweetnlp.load_model('sentiment')

    # annotate training data
    outf = open("../../data/tweetEval/train_sentiment.txt", "w")
    for i in range(len(train_tweets)):
        predict = model.sentiment(train_tweets[i])
        outf.write(predict['label'] + "\n")

    outf.close()

    # annotate test data
    outf = open("../../data/tweetEval/test_sentiment.txt", "w")
    for i in range(len(test_tweets)):
        predict = model.sentiment(test_tweets[i])
        outf.write(predict['label'] + "\n")

    outf.close()
