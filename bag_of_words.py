from read_tsv import get_saved_data
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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


def generate_vector(train, dict):
    dict_len = len(dict)
    vector_and_sentiment = []
    print(dict_len)
    train = train[:100]
    word_list = []
    for phrase in train:
        words = phrase[0]
        word_list.append(words)
    vectorizer = CountVectorizer()
    print(vectorizer.fit_transform(word_list).todense())
    print(vectorizer.vocabulary_)
    print(type(vectorizer.fit_transform(word_list).todense()))
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


train, test = get_saved_data()
generate_vector(train, get_dict())
