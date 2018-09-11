import csv
import pickle


def load_tsv():
    data = []
    with open('train.tsv') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            # 'PhraseId', 'SentenceId', 'Phrase', 'Sentiment'
            data.append([row[2], row[3]])
    return data


def split_data(data, ratio):
    phrase_count = len(data)
    train = data[1: int(phrase_count * ratio)]
    test = data[int(phrase_count * ratio):]
    return train, test


def save_data(train, test):
    with open("train.txt", "wb") as fp:
        pickle.dump(train, fp)
    with open("test.txt", "wb") as fp:
        pickle.dump(test, fp)


def get_saved_data():
    with open("train.txt", "rb") as fp:
        train = pickle.load(fp)
    with open("test.txt", "rb") as fp:
        test = pickle.load(fp)
    return train, test

# train, test = split_data(load_tsv(), 0.9)
# save_data(train, test)