
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
import numpy as np
from nltk.corpus import stopwords
import re

def generate_prompt(data_point):
    return f"""
            Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative".

            [{data_point["text"]}] = {data_point["sentiment"]}
            """.strip()

def generate_test_prompt(data_point):
    return f"""
            Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative".

            [{data_point["text"]}] = """.strip()
            
def generate_text(text):
    return f"""
            Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative".

            [{text}] = """.strip()

def remove_links(text):
    '''Ex:- https://www.google.com.eg/'''
    return re.sub("(http[s]?:\S+)","", text)

def remove_shortwords(text):
    text=re.sub("'", "", text)
    text = text.split()
    clean_text = [word for word in text if  len(word) > 1]
    return " ".join(clean_text)


def remove_mentions(text):
    '''@User Mention'''
    return re.sub("@[A-Za-z0-9_]+","", text)


def remove_stopwords(text):
    stopwords_list = stopwords.words('english')
    text = text.split()
    clean_text = [word for word in text if (word not in stopwords_list)]
    return " ".join(clean_text)

def remove_nonwords(text):
    text = re.sub('[^\w]',' ',text)
    return text

def clean_text(text):
    """
    Cleans the input text by performing the following operations in sequence:
    1. Removes links.
    2. Removes mentions.
    3. Converts text to lowercase.
    4. Removes stopwords.
    5. Removes non-word characters.
    6. Removes short words.
    7. Strips leading and trailing whitespace.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = remove_links(text)
    text = remove_mentions(text)
    text = text.lower()
    text = remove_stopwords(text)
    text = remove_nonwords(text)
    text = remove_shortwords(text)
    text = text.strip()
    return text

def tokenize_pad_sequences(textes, tokenizer, maxlen):
    '''
    This function tokenize the input text into sequnences of intergers and then
    pad each sequence to the same length
    '''
    X_array = []
    for text in textes:
        # Text tokenization
        X = tokenizer(text, padding="max_length", truncation=True, max_length = maxlen)
        # Pad sequences to the same length
        X = np.asarray(X["input_ids"]).reshape((1,-1))
        if len(X_array) == 0:
            X_array = X
        else:
            X_array = np.concatenate((X_array,X), axis = 0)
    # return sequences
    return X_array

SENTIMENT_LIST = ["positive", "neutral", "negative"]
DATA_SPLIT_RANDOM_STATE = 42
SAMPLE_RANDOM_STATE = 10
TRAIN_SIZE = 300
TEST_SIZE = 300
MAX_WORDS = 10000

class DATA_PROCESSOR():
    """
    DATA_PROCESSOR is a class designed to handle data processing tasks for financial news classification. 
    It includes methods for initializing the processor, reading data from a CSV file, and performing various 
    data preprocessing steps such as tokenization, text cleaning, and prompt generation.
    Attributes:
        tokenizer (BertTokenizerFast): Tokenizer for processing text data.
        maxlen (int): Maximum length for tokenized sequences.
        clean_mode (bool): Flag to indicate whether text cleaning should be performed.
        prompt_mode (bool): Flag to indicate whether prompt generation should be performed.
        tokenize_mode (bool): Flag to indicate whether tokenization should be performed.
        X_train (list): List to store training data.
        X_test (list): List to store test data.
        Y_train (list): List to store training labels.
        Y_test (list): List to store test labels.
        X_eval (list): List to store evaluation data.
        Y_eval (list): List to store evaluation labels.
        y_test_true (list): List to store true test labels.
    Methods:
        __init__(tokenizer_name: str = "", **kwargs): Initializes the DATA_PROCESSOR with optional tokenizer name and other parameters.
        reinit(): Reinitializes the data attributes.
        read_csv(url): Reads data from a CSV file, processes it, and splits it into training, test, and evaluation sets.
    """
    def __init__(self, tokenizer_name : str = "", **kwargs):
        self.reinit()
        checkpoint = "bert-base-uncased"
        if tokenizer_name !="":
            checkpoint = tokenizer_name
        self.tokenizer = BertTokenizerFast.from_pretrained(checkpoint)

        self.maxlen = kwargs.get("maxlen") if kwargs.get("maxlen") else 128
        self.clean_mode = kwargs.get("clean_mode") if kwargs.get("clean_mode") else True
        self.prompt_mode = kwargs.get("prompt_mode") if kwargs.get("prompt_mode") else False
        self.tokenize_mode = kwargs.get("tokenize_mode") if kwargs.get("tokenize_mode") else False
    
    def reinit(self):
        self.X_train = list()
        self.X_test = list()
        self.Y_train = list()
        self.Y_test = list()
        self.X_eval = list()
        self.Y_eval = list()
        self.X_train_org = list()
        self.X_eval_org = list()
        self.X_test_org = list()
        self.y_test_true = None
        
    def read_csv(self, url):
        self.reinit
        df = pd.read_csv(url, 
                 names=["sentiment", "text"],
                 encoding="utf-8", encoding_errors="replace")
        df_y = pd.get_dummies(df.sentiment, dtype=int)
        for sentiment in SENTIMENT_LIST:
            train, test, train_target, test_target  = train_test_split(df[df.sentiment==sentiment], df_y[df.sentiment==sentiment],
                                            train_size=TRAIN_SIZE,
                                            test_size=TEST_SIZE, 
                                            random_state=DATA_SPLIT_RANDOM_STATE)
            self.X_train.append(train)
            self.X_test.append(test)
            self.Y_train.append(train_target)
            self.Y_test.append(test_target)
        self.X_train = pd.concat(self.X_train).sample(frac=1, random_state=SAMPLE_RANDOM_STATE)
        self.X_test = pd.concat(self.X_test)
        self.Y_train = pd.concat(self.Y_train).loc[self.X_train.index]
        self.Y_test = pd.concat(self.Y_test)
        
        self.X_train_org = self.X_train
        self.X_test_org = self.X_test

        eval_idx = [idx for idx in df.index if idx not in list(self.X_train.index) + list(self.X_test.index)]
        self.X_eval = df[df.index.isin(eval_idx)]

        if self.prompt_mode:
            self.X_eval = (self.X_eval
                .groupby('sentiment', group_keys=False)
                .apply(lambda x: x.sample(n=50, random_state=SAMPLE_RANDOM_STATE, replace=True)))
        else:
            self.X_eval = (self.X_eval
                    .groupby('sentiment', group_keys=False)
                    .apply(lambda x: x.sample(n=50, random_state=SAMPLE_RANDOM_STATE, replace=True)))

        self.Y_eval = df_y.loc[self.X_eval.index]
        
        self.X_train = self.X_train.reset_index(drop=True)
        
        self.y_test_true = self.X_test.sentiment
        
        if self.clean_mode:
            self.X_train["text"] = self.X_train["text"].apply(clean_text)
            self.X_eval["text"] = self.X_eval["text"].apply(clean_text)
            self.X_test["text"] = self.X_test["text"].apply(clean_text)
            
        if self.prompt_mode:
            self.X_train = pd.DataFrame(self.X_train.apply(generate_prompt, axis=1), 
                       columns=["text"])
            self.X_eval = pd.DataFrame(self.X_eval.apply(generate_prompt, axis=1), 
                      columns=["text"])
            
            self.X_test = pd.DataFrame(self.X_test.apply(generate_test_prompt, axis=1), columns=["text"])

        if self.tokenize_mode:
            self.X_train = tokenize_pad_sequences(self.X_train["text"], self.tokenizer, self.maxlen)
            self.X_eval = tokenize_pad_sequences(self.X_eval["text"], self.tokenizer, self.maxlen)
            self.X_test = tokenize_pad_sequences(self.X_test["text"], self.tokenizer, self.maxlen)
            








