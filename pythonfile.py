import pickle
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

# Load NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load and preprocess the dataset
def preprocess_data(file_path):
    data = pd.read_csv(file_path, header=None, nrows=1000000)
    data.dropna(subset=[1], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data[1] = data[1].astype(str) + ' ' + data[2].astype(str)
    data = data.drop(data.columns[2], axis=1)

    # Text preprocessing steps (remove HTML tags, URLs, stopwords, lemmatization, etc.)
    def remove_tags(string):
        removelist = ""
        result = re.sub('<.*?>', '', string)          # Remove HTML tags
        result = re.sub(r'https?://\S+|www\.\S+', '', result)  # Remove URLs
        result = re.sub(r'\W+', ' ', result)    # Remove non-alphanumeric characters
        result = result.lower()
        return result

    data[1] = data[1].apply(lambda cw : remove_tags(cw))

    stop_words = set(stopwords.words('english'))
    data[1] = data[1].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(text):
        st = ""
        for w in w_tokenizer.tokenize(text):
            st = st + lemmatizer.lemmatize(w) + " "
        return st

    data[1] = data[1].apply(lemmatize_text)
    
    return data

# Split dataset into training and test sets
def split_data(data, train_size):
    total_samples = len(data)
    train_samples = int(total_samples * (train_size / 100))
    train_set = data[:train_samples]
    test_set = data[-int(total_samples * 0.2):]  # Last 20% of samples
    return train_set, test_set

def default_value():
    return 1 # Apply add-1 smoothing

# Train Naive Bayes classifier and store parameters
def train_naive_bayes(train_data, default_value_func):
    word_counts = {}
    class_counts = {}
    classes = train_data[0].unique()
    vocab = set()
    
    # Initialize dictionary to store parameters
    classifier_params = {
        'word_counts': {},
        'class_counts': {},
        'vocab': set()
    }
    
    for c in classes:
        class_data = train_data[train_data[0] == c][1]
        class_counts[c] = len(class_data)
        word_counts[c] = defaultdict(default_value_func)

        for sentence in class_data:
            words = set(nltk.word_tokenize(sentence))  # Use set to get unique words
            for word in words:
                word_counts[c][word] += 1
                vocab.add(word)
                
        # Store parameters for the class in the dictionary
        classifier_params['word_counts'][c] = word_counts[c]
        classifier_params['class_counts'][c] = class_counts[c]
        classifier_params['vocab'].update(vocab)
        
    return classifier_params

# Test Naive Bayes classifier using stored parameters
def test_naive_bayes(test_data, classifier_params):
    word_counts = classifier_params['word_counts']
    class_counts = classifier_params['class_counts']
    vocab = classifier_params['vocab']
    
    predictions = []
    true_labels = test_data[0].tolist()
    for _, sentence in test_data.iterrows():
        max_prob = -math.inf
        argmax_class = None
        for c, count in class_counts.items():
            prob = math.log(count / len(test_data))
            for word in set(nltk.word_tokenize(sentence[1])):  # Use set to get unique words
                if word in vocab:
                    prob += math.log((word_counts[c][word]) / (class_counts[c] + len(vocab)))  
            if prob > max_prob:
                max_prob = prob
                argmax_class = c
        predictions.append(argmax_class)
    accuracy = sum(1 for x, y in zip(predictions, true_labels) if x == y) / len(predictions)
    return accuracy, true_labels, predictions

# Classify user-entered sentence using the trained classifier parameters
def classify_user_sentence(sentence, classifier_params):
    word_counts = classifier_params['word_counts']
    class_counts = classifier_params['class_counts']
    vocab = classifier_params['vocab']
    
    # Calculate probabilities for each class label
    class_probabilities = {}
    for c, count in class_counts.items():
        log_prob = math.log(count) - math.log(len(classifier_params['test_data']))
        for word in nltk.word_tokenize(sentence):
            if word in vocab:
                log_prob += math.log((word_counts[c][word] + 1) / (class_counts[c] + len(vocab)))
        # Convert log probability back to linear space
        class_probabilities[c] = math.exp(log_prob)
    
    return class_probabilities

# Main function
def main(train_size):
    print("Kaur, Manpreet solution:")
    print(f"Training set size: {train_size} %")
    print()
    # Load and preprocess the dataset
    data = preprocess_data("C:/Users/kaur6/Downloads/Senitment_Analysis/Kaur_Manpreet_CS585_Programming02/amazon_reviews.csv")

    # Split dataset into training and test sets
    train_data, test_data = split_data(data, train_size)
    
    # Train Naive Bayes classifier and store parameters
    print("Training classifier...")
    classifier_params = train_naive_bayes(train_data, default_value)
    classifier_params['test_data'] = test_data  # Store test data for log-space calculation

    # Test Naive Bayes classifier using stored parameters
    print("Testing classifier...")
    accuracy, true_labels, predictions = test_naive_bayes(test_data, classifier_params)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    # Print results
    print("Test results / metrics:")
    print()
    print(f"Number of true positives: {tp}")
    print(f"Number of true negatives: {tn}")
    print(f"Number of false positives: {fp}")
    print(f"Number of false negatives: {fn}")
    print(f"Sensitivity (recall): {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Precision: {precision}")
    print(f"Negative predictive value: {npv}")
    print(f"Accuracy: {accuracy}")
    print(f"F-score: {f1_score}")
    print()
    
    # Ask the user for keyboard input
    while True:
        sentence = input("Enter your sentence:\n")
        print()
        if sentence.strip() == "":
            break
        class_probabilities = classify_user_sentence(sentence, classifier_params)
        print(f"Sentence S:")
        print()
        print(f"{sentence}")
        print()
        max_prob_class = max(class_probabilities, key=class_probabilities.get)
        print(f"was classified as Label {max_prob_class}.")
        for class_label, probability in class_probabilities.items():
            print(f"P(Label {class_label} | S) = {probability}")
        print()
        choice = input("Do you want to enter another sentence [Y/N]? ")
        if choice.upper() != 'Y':
            break

if __name__ == "__main__":

    import sys

    # Check command-line arguments
    if len(sys.argv) != 2:
        # More than one argument provided or no argument provided
        train_size = 80
    else:
        try:
            train_size = int(sys.argv[1])
            if train_size < 20 or train_size > 80:
                train_size = 80
        except ValueError:
            # Argument provided is not a valid integer
            train_size = 80

    main(train_size)

