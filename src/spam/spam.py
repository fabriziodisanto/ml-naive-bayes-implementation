import numpy as np

from src.spam import util, svm


def get_words(message: str) -> [str]:
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For simplicity, you should split on whitespace, not
    punctuation or any other character. For normalization, you should convert
    everything to lowercase.  Please do not consider the empty string (" ") to be a word.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """
    return message.lower().split()


def create_dictionary(messages: [str]) -> {str, int}:
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    indices_dictionary: {str, int} = dict()
    number_of_messages: {str, (str, int)} = dict()
    index = 0
    for message in messages:
        for word in get_words(message):
            if word not in number_of_messages.keys():
                number_of_messages[word] = (message, 1)
            else:
                if number_of_messages[word][0] != message:
                    if number_of_messages[word][1] == 4:
                        if word not in indices_dictionary.keys():
                            indices_dictionary[word] = index
                            index += 1
                    else:
                        number_of_messages[word] = (message, number_of_messages[word][1] + 1)
    return indices_dictionary


def transform_text(messages: [str], word_dictionary: {str, int}):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    word_keys: [str] = list(word_dictionary.keys())
    matrix = np.zeros(shape=(len(messages), len(word_keys)), dtype=np.int8)
    for message_index, message in enumerate(messages):
        for word in get_words(message):
            if word in word_keys:
                matrix[message_index][word_keys.index(word)] += 1
    return matrix


def count_total_words(matrix) -> {int, int}:
    words_count: {int, int} = dict()
    transpose = matrix.transpose()
    for index, word_entry in enumerate(transpose):
        words_count[index] = sum(word_entry)
    return words_count


def get_this_word_probability(matrix, labels, word_index, spam_words_count, not_spam_words_count):
    # Each row in the resulting array should correspond to each message
    # and each column should correspond to a word of the vocabulary.
    # p(x word | spam) = p(spam ^ x word) / p(spam)
    #               ['word1' 'word2' 'word3']  [label]
    # ['message 1']    0       2        0         1
    # ['message 2']    1       1        3         0
    # ['message 3']    1       0        1         1
    # word 1 spam = 1/4
    # word 1 not spam = 1/5
    # word 2 spam = 2/4
    # word 2 not spam = 1/5
    # word 3 spam = 1/4
    # word 3 not spam = 3/5

    spam_and_this_word = 0
    not_spam_and_this_word = 0

    for index, message in enumerate(matrix):
        value = message[word_index]
        if value > 0:
            if labels[index] == 1:
                spam_and_this_word += value
            else:
                not_spam_and_this_word += value

    this_word_given_spam_probability = spam_and_this_word / spam_words_count
    this_word_given_not_spam_probability = not_spam_and_this_word / not_spam_words_count

    return this_word_given_spam_probability, this_word_given_not_spam_probability


def get_words_count_filtered_by_spams(matrix, labels):
    spam_words_count = 0
    not_spam_words_count = 0
    for message_index in range(len(labels)):
        if labels[message_index] == 1:
            spam_words_count += sum(matrix[message_index])
        else:
            not_spam_words_count += sum(matrix[message_index])
    return spam_words_count, not_spam_words_count


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """
    # P(spam | uso 'x') = P(uso 'x' | spam) * P(spam) / P('x')
    # merged = np.c_[matrix, labels]
    words = np.zeros(shape=(matrix.shape[1], 2), dtype=np.float64)
    words_count_dict: {int, int} = count_total_words(matrix)

    spam_words_count, not_spam_words_count = get_words_count_filtered_by_spams(matrix, labels)
    spam = np.array([np.count_nonzero(labels == 1)/len(labels), np.count_nonzero(labels == 0)/len(labels)], dtype=np.float32)

    for i in range(matrix.shape[1]):
        this_word_count = words_count_dict[i]
        # if this_word_count == 0:
        #     words[i][0] = 0
        #     words[i][1] = 0
        # else:
        this_word_and_spam_probability, this_word_and_not_spam_probability = get_this_word_probability(
            matrix, labels, i, spam_words_count, not_spam_words_count)
        words[i][0] = this_word_and_spam_probability
        words[i][1] = this_word_and_not_spam_probability

    return dict(spam=spam, words=words)


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    #               ['word1' 'word2' 'word3']  [label]
    # ['message 1']    0       2        0         1
    # ['message 2']    1       1        3         0
    # ['message 3']    1       0        1         1

    predict = np.zeros(shape=matrix.shape[0], dtype=np.int8)

    for message_index, message in enumerate(matrix):
        # message_is_spam_probability = math.log(model['spam'][0])
        message_is_spam_probability = model['spam'][0]
        message_is_not_spam_probability = model['spam'][1]
        # message_is_not_spam_probability = math.log(model['spam'][1])
        for word_index, count in enumerate(message):
            if count > 0:
                # if model['words'][word_index][0] > 0:
                # message_is_spam_probability += (math.log(model['words'][word_index][0]) * count)
                message_is_spam_probability *= (model['words'][word_index][0] ** count)
                # message_is_spam_probability *= (model['words'][word_index][0])

                # if model['words'][word_index][1] > 0:
                # message_is_not_spam_probability += (math.log(model['words'][word_index][1]) * count)
                message_is_not_spam_probability *= (model['words'][word_index][1] ** count)
                # message_is_not_spam_probability *= (model['words'][word_index][1])

        # else:
        predict[message_index] = 1 if message_is_spam_probability > message_is_not_spam_probability else 0
    return predict


def get_word_with_this_index(dictionary, value):
    for k, v in dictionary.items():
        if v == value:
            return k
    return None


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    most_indicative_words = []
    words = model['words'].transpose()
    spam_probabilities = words[0]
    non_spam_probabilities = words[1]
    probabilities = np.zeros(shape=(spam_probabilities.shape[0]), dtype=np.float32)
    for index in range(probabilities.shape[0]):
        probabilities[index] = spam_probabilities[index] / (spam_probabilities[index] + non_spam_probabilities[index])
    values = np.argsort(probabilities)[len(probabilities) - 5:]
    for value in values:
        most_indicative_words.append(get_word_with_this_index(dictionary, value))
    return most_indicative_words


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """

    best_accuracy = 0
    best_radius = 0

    for radius in radius_to_consider:
        predictions = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = np.mean(predictions == val_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_radius = radius
        print("Radius {radius} done. Accuracy was {accuracy}".format(radius=radius, accuracy=accuracy))
    return best_radius


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('UAML2020_spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('UAML2020_spam_sample_train_matrix', train_matrix[:100, :])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('UAML2020_spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('UAML2020_spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.001, 0.01, 0.1, 1, 10])
    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels,
                                             [optimal_radius * 0.5, optimal_radius * 0.75, optimal_radius,
                                              optimal_radius * 1.25, optimal_radius * 1.5])

    util.write_json('UAML2020_spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
