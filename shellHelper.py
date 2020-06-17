import os
import csv
from datetime import date

def run_sentiment_training():
    os.system("ls")
    os.system("nohup python amex/services/customSentiment.py > output.log &")
    print("done")

def edit_sentiment_csv(df, replace=False):
    for i in range(len(df)):
        with open(os.path.join(os.path.dirname( __file__ ), '../services/sentiments.csv'), "w" if replace else "a", newline='\n') as file:
            writer = csv.writer(file)

            if replace:
                replace = False
                writer.writerow(['Emotion', 'Review', 'Time'])
            enterData = [df['emotion'][i], df['sentence'][i], date.today()]
            writer.writerow(enterData)

    return (str(i+1) + " Rows updated.")