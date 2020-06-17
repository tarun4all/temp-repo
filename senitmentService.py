import numpy as np
import re
import pickle
import nltk
import os
from nltk.corpus import stopwords
from sklearn.datasets import load_files

# nltk.download("stopwords")

# reviews = load_files('txt_sentoken/')
# X,y = reviews.data, reviews.target

# #storing into pickle
# with open('X.pickle', 'wb') as f:
#     pickle.dump(X, f)

# with open('y.pickle', 'wb') as f:
#     pickle.dump(y, f)

# #unplickling code
# with open('X.pickle', 'rb') as f:
#     X = pickle.load(f)

# with open('y.pickle', 'rb') as f:
#     y = pickle.load(f)

# #pre processing over data
# corpus = []
# for i in range(0, len(X)):
#     review = re.sub(r'\W', ' ', str(X[i]))
#     review = review.lower()
#     review = re.sub(r'\s+[a-z]\s+', ' ', review)
#     review = re.sub(r'^[a-z]\s+', ' ', review)
#     review = re.sub(r'\s+', ' ', review)
#     corpus.append(review)

# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer =  CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
# X = vectorizer.fit_transform(corpus).toarray()

# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer =  TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
# X = vectorizer.fit_transform(corpus).toarray()

# from sklearn.feature_extraction.text import TfidfTransformer
# transformer = TfidfTransformer()
# X = transformer.fit_transform(X).toarray()

# #train model
# from sklearn.model_selection import train_test_split

# text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# # random state for same result each time

# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression()
# classifier.fit(text_train, sent_train)

# #test model to check
# sent_pred = classifier.predict(text_test)

# #to check validity
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(sent_test, sent_pred)

# #to get rate 
# result = cm[0][0] + cm[1][1]
# # print(result)

# # save the model for future use in pickle file
# with open('classifier.pickle', 'wb') as f:
#     pickle.dump(classifier, f)

# #pickling the vectorizer
# with open('tfidf.pickle', 'wb') as f:
#     pickle.dump(vectorizer, f)

# # to check we have to give data with vectorizer
# a = classifier.predict("I am happy")
# print(a)

#unpickling the classifier and vectorizer
with open(os.path.join(os.path.dirname( __file__ ), 'classifier.pickle'), 'rb') as f:
    clf = pickle.load(f)

with open(os.path.join(os.path.dirname( __file__ ), 'tfidf.pickle'), 'rb') as f:
    tfidf = pickle.load(f)


def predictEmotion(sent):
    sample = [sent]
    sample = tfidf.transform(sample).toarray()
    return (clf.predict(sample))[0]