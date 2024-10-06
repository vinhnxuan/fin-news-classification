import numpy as np
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate(y_true, y_pred, method, show = False):
    """
    Evaluate the performance of a classification model.
    Parameters:
    y_true (list or array-like): True labels.
    y_pred (list or array-like): Predicted labels.
    method (str): The method name or identifier used for saving plots.
    Returns:
    None
    This function performs the following evaluations:
    1. Calculates and prints the overall accuracy.
    2. Calculates and prints the accuracy for each unique label.
    3. Generates and prints a classification report.
    4. Plots and saves the classification report as a heatmap.
    5. Generates and prints a confusion matrix.
    6. Plots and saves the confusion matrix as a heatmap.
    The plots are saved with filenames based on the provided method identifier.
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    
    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels
    
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')
        
    # Generate classification report
    
    class_report = classification_report(y_true=y_true, y_pred=y_pred, output_dict= True)
    class_report_df = pd.DataFrame(class_report).transpose()
    print('\nClassification Report:')
    # Plot classification report
    plt.figure(figsize=(10, 6))
    sns.heatmap(class_report_df.iloc[:-1, :].T, annot=True, cmap='Blues', cbar=False)
    plt.title('Classification Report')
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.savefig(method + "_class_report.png")
    if show:
        plt.show()
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(method + "_confusion_matrix.png")
    if show:
        plt.show()
    print(conf_matrix)

from nltk.tokenize import TweetTokenizer
def count_tweets(result, tweets, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    for y, tweet in zip(ys, tweets):
        tweet_tokens = tokenizer.tokenize(tweet)
        for word in tweet_tokens:
            # define the key, which is the word and label tuple
            pair = (word, y)

            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1

            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1
    ### END CODE HERE ###

    return result