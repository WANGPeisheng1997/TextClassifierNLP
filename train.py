from read_and_write import get_saved_data, load_test, write_csv
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from logistic_regression import logistic_regression
import time

def generate_dict(train):
    dict = []
    for phrase in train:
        words = phrase[0].split()
        for word in words:
            if word not in dict:
                dict.append(word)
    with open("word_dict.txt", "wb") as fp:
        pickle.dump(dict, fp)


def get_dict():
    with open("word_dict.txt", "rb") as fp:
        dict = pickle.load(fp)
    return dict


def generate_vector_and_vocabulary(train):
    sentiment = []
    word_list = []
    for phrase in train:
        words = phrase[0]
        word_list.append(words)
    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(word_list).toarray()
    vocabulary = vectorizer.vocabulary_

    for sample in train:
        sentiment.append(int(sample[1]))

    return vector, sentiment, vocabulary


def generate_vector_and_vocabulary_ngram(train):
    sentiment = []
    word_list = []
    for phrase in train:
        words = phrase[0]
        word_list.append(words)
    vectorizer = CountVectorizer(ngram_range=(1,3))
    vector = vectorizer.fit_transform(word_list).toarray()
    vocabulary = vectorizer.vocabulary_

    for sample in train:
        sentiment.append(int(sample[1]))

    return vector, sentiment, vocabulary


def generate_classifier(vec, sentiment, learning_rate, iterate_times, threshold, batch_size): # >= threshold becomes 1 and others 0
    new_class = []
    for senti in sentiment:
        if senti >= threshold:
            new_class.append(1)
        else:
            new_class.append(0)
    w = logistic_regression(vec, new_class, learning_rate, iterate_times, batch_size)
    return w


def save_all_classifier_results(vec, sentiment, learning_rate, iterate_times, batch_size):
    result_w = []
    for i in [1, 2, 3, 4]:
        result_w.append(generate_classifier(vec, sentiment, learning_rate, iterate_times, i, batch_size))
    with open("classifier_lr_"+str(learning_rate)+"_epoch_"+str(iterate_times)+".txt", "wb") as fp:
        pickle.dump(result_w, fp)
    return result_w


def get_classifiers(path):
    with open(path, "rb") as fp:
        classifiers = pickle.load(fp)
    return classifiers


def test_classfier(test, vocabulary, classifiers):
    vectorizer = CountVectorizer()
    correct = 0
    count = [0,0,0,0,0]
    for phrase in test:
        f = []
        words = phrase[0]
        vector = [0] * (len(classifiers[0])-1)
        tokens = vectorizer.build_tokenizer()(words)
        for token in tokens:
            if token in vocabulary:
                vector[vocabulary[token]] += 1
        x = np.array([1] + vector)
        for classifier in classifiers:
            f.append(x.dot(classifier))

        estimate_phrase_class = 0
        for i in range(len(f)):
            if f[i] > 0:
                estimate_phrase_class = i + 1

        count[estimate_phrase_class] += 1

        true_phrase_class = int(phrase[1])

        if(estimate_phrase_class == true_phrase_class):
            correct += 1

    print("Correct: " + str(correct) + "/" + str(len(test)))
    print(correct/len(test))
    print(count)


def generate_result_from_classifier(vocabulary, classifiers):
    test = load_test()
    result = []
    vectorizer = CountVectorizer()
    for phrase_id, phrase in test:
        f = []
        vector = [0] * (len(classifiers[0]) - 1)
        tokens = vectorizer.build_tokenizer()(phrase)
        for token in tokens:
            if token in vocabulary:
                vector[vocabulary[token]] += 1
        x = np.array([1] + vector) # the word vector
        for classifier in classifiers:
            f.append(x.dot(classifier))

        estimate_phrase_class = 0
        for i in range(len(f)):
            if f[i] > 0:
                estimate_phrase_class = i + 1

        result.append([phrase_id, estimate_phrase_class])

    return result


def test_classfier_ngram(test, vocabulary, classifiers):
    vectorizer = CountVectorizer(ngram_range=(1,3))
    correct = 0
    count = [0,0,0,0,0]
    for phrase in test:
        f = []
        words = phrase[0]
        vector = [0] * (len(classifiers[0])-1)
        tokens = vectorizer.build_analyzer()(words)
        for token in tokens:
            if token in vocabulary:
                vector[vocabulary[token]] += 1
        x = np.array([1] + vector)
        for classifier in classifiers:
            f.append(x.dot(classifier))

        estimate_phrase_class = 0
        for i in range(len(f)):
            if f[i] > 0:
                estimate_phrase_class = i + 1

        count[estimate_phrase_class] += 1

        true_phrase_class = int(phrase[1])

        if(estimate_phrase_class == true_phrase_class):
            correct += 1

    print("Correct: " + str(correct) + "/" + str(len(test)))
    print(correct/len(test))
    print(count)


def bag_of_words(learning_rate, epochs, batch_size):
    start = time.clock()
    train, test = get_saved_data()
    vec, sentiment, vocabulary = generate_vector_and_vocabulary(train)
    classifiers = save_all_classifier_results(vec, sentiment, learning_rate, epochs, batch_size)
    test_classfier(test, vocabulary, classifiers)
    count = [0, 0, 0, 0, 0]
    for phrase in test:
        count[int(phrase[1])] += 1
    print(count)
    end = time.clock()-start
    print("Learning rate: " + str(learning_rate) + " Epoch: " + str(epochs) + " Batch size: " + str(batch_size))
    print("Time: " + str(end))
    return vocabulary, classifiers


def ngram(learning_rate, epochs, batch_size):
    start = time.clock()
    train, test = get_saved_data()
    vec, sentiment, vocabulary = generate_vector_and_vocabulary_ngram(train[:30000])
    classifiers = save_all_classifier_results(vec, sentiment, learning_rate, epochs, batch_size)
    test_classfier_ngram(test, vocabulary, classifiers)
    count = [0, 0, 0, 0, 0]
    for phrase in test:
        count[int(phrase[1])] += 1
    print(count)
    end = time.clock()-start
    print("Learning rate: " + str(learning_rate) + " Epoch: " + str(epochs) + " Batch size: " + str(batch_size))
    print("Time: " + str(end))
    return vocabulary, classifiers