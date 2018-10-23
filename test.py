from train import *

def test(learning_rate, epoch, batch_size, is_Ngram=False, generate_csv=False):
    # is_Ngram: use Bag-of-words model or Ngram model
    # generate_csv: only test on the validation set or generate the csv file for submission
    if is_Ngram:
        vocabulary, classifier = ngram(learning_rate, epoch, batch_size)
        if generate_csv:
            result = generate_result_from_classifier(vocabulary, classifier)
            write_csv(result, "ngram_lr_" + str(learning_rate) + "_epoch_" + str(epoch) + ".csv")
    else:
        vocabulary, classifier = bag_of_words(learning_rate, epoch, batch_size)
        if generate_csv:
            result = generate_result_from_classifier(vocabulary, classifier)
            write_csv(result, "bow_lr_" + str(learning_rate) + "_epoch_" + str(epoch) + ".csv")


test(learning_rate=5, epoch=5, batch_size=64, is_Ngram=False, generate_csv=True)
