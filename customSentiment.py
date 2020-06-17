import pickle
import pandas as pd
import numpy as np
import os
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def train_sentiment():
    print("training start")
    df = pd.read_csv(os.path.join(os.path.dirname( __file__ ), 'sentiments.csv'))
    df.head()

    col = ['Emotion', 'Review']
    df = df[col]
    df = df[pd.notnull(df['Review'])]
    df.columns = ['Emotion', 'Review']
    df['category_id'] = df['Emotion'].factorize()[0]
    category_id_df = df[['Emotion', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Emotion']].values)
    df.head()

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.Review).toarray()
    labels = df.category_id
    features.shape

    N = 2
    for Product, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}':".format(Product))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

    X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Emotion'], random_state = 0)

    #vector initialise
    count_vect = CountVectorizer()

    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    # Save the vectorizer
    vec_file = 'amex/services/vectorizer.pickle'
    pickle.dump(count_vect, open(vec_file, 'wb'))

    with open('amex/services/sentiment_custom_classifier', 'wb') as picklefile:
        pickle.dump(clf,picklefile)

    return True

def predict_custom_sentiment(para):
    loaded_vectorizer = pickle.load(open('amex/services/vectorizer.pickle', 'rb'))

    with open('amex/services/sentiment_custom_classifier', 'rb') as training_model:
        model = pickle.load(training_model)

    probability = (model.predict_proba(loaded_vectorizer.transform([para])))
    emotion = (model.predict(loaded_vectorizer.transform([para])))

    sentimentAnalysis = (probability[0][0] if emotion[0] == "negative" else probability[0][1])
    return [emotion[0] + "-" + str(sentimentAnalysis)]

if __name__ == "__main__":
    train_sentiment()