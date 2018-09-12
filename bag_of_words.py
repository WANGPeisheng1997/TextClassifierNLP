from read_tsv import get_saved_data
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from logistic_regression import logistic_regression

def generate_dict(train):
    dict = []
    for phrase in train:
        words = phrase[0].split()
        for word in words:
            if word not in dict:
                dict.append(word)
    with open("word_dict.txt", "wb") as fp:
        pickle.dump(dict, fp)

# train = split_data(load_tsv(), 0.9)[0]
# generate_dict(train)


def get_dict():
    with open("word_dict.txt", "rb") as fp:
        dict = pickle.load(fp)
    return dict


def generate_vector(train):
    train = train[:50000]
    vector_and_sentiment = []
    word_list = []
    for phrase in train:
        words = phrase[0]
        word_list.append(words)
    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(word_list).toarray()
    for i in range(len(train)):
        if(int(train[i][1])) > 0:
            vector_and_sentiment.append([vector[i], 1])
        else:
            vector_and_sentiment.append([vector[i], 0])

    return vector_and_sentiment

    # print(vectorizer.vocabulary_)

    # for phrase in train:
    #     words = phrase[0].split()
    #     vector = [0] * dict_len
    #     for word in words:
    #         index = dict.index(word)
    #         vector[index] = 1
    #     vector_and_sentiment.append([vector, phrase[1]])
    # with open("phrase_vector.txt", "wb") as fp:
    #     pickle.dump(vector_and_sentiment, fp)
    # print(vector_and_sentiment)
    # return vectorizer.fit_transform(word_list).todense()

train, test = get_saved_data()

vec = generate_vector(train)
logistic_regression(vec, 0.5, 1)
