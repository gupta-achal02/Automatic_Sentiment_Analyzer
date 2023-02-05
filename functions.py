import torch
from transformers import AutoTokenizer
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import fnmatch
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

analyser = SentimentIntensityAnalyzer()
lemma = WordNetLemmatizer() 
stop_words = stopwords.words("english")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = torch.load('finetuned_bert_model')
model.to('cpu')

class_labels = ['Negative', 'Positive']
model.config.id2label = class_labels
sentiment_model = pipeline(task='sentiment-analysis', model=model, tokenizer=tokenizer)

def get_df(file_path, col):
    """
    Creates a DataFrame from the input csv file

    Args:
        file_path (str): Path of the input csv file
        col (str): Name of the column to be used for sentiment analysis

    Returns:
        pandas.DataFrame: DataFrame created from the input csv file
    """

    # reading the input csv file
    df = pd.read_csv(f"{file_path}.csv")
    df.rename(columns = lambda x: x.strip(), inplace = True)
    # Dropping duplicated rows
    df = df.drop_duplicates(subset = col, keep = "first")
    # Drop rows with missing values
    df = df.dropna(subset = [col])
    # Resetting the index of the dataframe
    df = df.reset_index(drop = True)
    return df


def get_sentiment_DistilBERT(text):
    """
    Gets the sentiment label for a string using DistilBERT model

    Args:
        text (str): A string to be analyzed

    Returns:
        str: Sentiment Label for the string
    """
    
    return sentiment_model(text)[0]["label"]

def get_sentiment_VADER(text):
    """
    Gets the sentiment label for a string using VADER sentiment analyzer

    Args:
        text (str): A string to be analyzed

    Returns:
        str: Sentiment Label for the string
    """

    polarity_dict = analyser.polarity_scores(text)
    return "Positive" if polarity_dict['compound'] >= 0 else "Negative"


def set_sentiments(df, col):
    """
    Assigns sentiment values to each record.
    Model to be used is selected based on the number of rows

    Args:
        df (pandas.DataFrame): Dataframe created from the input csv file
        col (str): Column name selected by user

    Returns:
        pandas.DataFrame: DataFrame updated with a new column
    """

    df["sentiment"] = df[col].apply(lambda x: get_sentiment_DistilBERT(x)) if df.shape[0] <= 1000 else df[col].apply(lambda x: get_sentiment_VADER(x))
    return df


def get_output(df, file_path):
    """
    Creates an output csv file with the updated dataframe

    Args:
        df (pandas.DataFrame): DataFrame updated with a new column
        file_path (str): File path for the output csv file
    """

    df.to_csv(f"{file_path}_output.csv", index = False)


def process_text(df, col):
    """
    Tokenizes, lemmatizes and removes stop words from the selected column    

    Args:
        df (pandas.DataFrame): Dataframe created from the input csv file
        col (str): Column name selected by user

    Returns:
        pandas.DataFrame: Updated DataFrame
    """

    # Converting all the texts into lowercase
    df[col] = df[col].apply(lambda x: x.lower())
    
    # Replacing urls with a space
    df[col] = df[col].apply(lambda x: re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', str(x)))
    
    # Replacing words starting with $ symbol with a space
    df[col] = df[col].apply(lambda x: re.sub('\$[a-zA-Z0-9]*', ' ', str(x)))
    
    # Replacing words starting with @ symbol with a space
    df[col] = df[col].apply(lambda x: re.sub('\@[a-zA-Z0-9]*', ' ', str(x)))
    
    # Replacing everything which doesn't contain a letter or an apostrophe with a space
    df[col] = df[col].apply(lambda x: re.sub('[^a-zA-Z\']', ' ', str(x)))
    
    # Removing words with single letters
    df[col] = df[col].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))
    
    # Tokenizing the texts
    df[col] = df[col].apply(lambda x: ' '.join([lemma.lemmatize(w) for w in nltk.wordpunct_tokenize(x) if w not in stop_words]))
    
    # Removing stop words
    df[col] = df[col].apply(lambda x: [lemma.lemmatize(w, nltk.corpus.reader.wordnet.VERB) for w in nltk.wordpunct_tokenize(x) if w not in stop_words])
    
    # Joining the tokenized words into whole texts
    df[col] = df[col].apply(lambda x: ' '.join(x))

    return df


def create_wordcloud(text, sentiment):
    """
    Creates wordcloud and saves it in static directory.

    Args:
        text (numpy.ndarray): Array of texts with the selected sentiment
        sentiment (str): Selected sentiment
    """

    # creating wordcloud
    words = ' '.join([words for words in text])
    wordcloud = WordCloud(width = 800, height = 400, max_font_size = 70, 
                          max_words = 100, background_color = "white").generate(words)
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation = "bilinear")
    plt.axis("off")
    # saving wordcloud in static directory
    plt.savefig(f"static/wordcloud_{sentiment}.png", bbox_inches = "tight", transparent = True)


def get_plots(df, col):
    """
    Creates countplot, piechart and wordcloud,
    and saves them in static directory.

    Args:
        df (pandas.DataFrame): Dataframe created from the input csv file
        col (str): Column name selected by user
    """

    cp = sns.countplot(x = df["sentiment"])
    fig1 = cp.get_figure()
    fig1.savefig("static/countplot.png", bbox_inches = "tight", transparent = True)
    plt.close(fig1)
    
    fig2 = plt.gcf()
    fig2.set_size_inches(7,7)

    colors = ["#66b3ff", "#ff9999"]
    plt.pie(df["sentiment"].value_counts(), labels = ("Positive", "Negative"), radius = 3, autopct = "%1.1f%%",
            shadow = True, startangle = 90, labeldistance = 1.1, colors = colors, explode = (0.1, 0.1))
    plt.axis("equal")
    plt.savefig("static/pie_chart.png", bbox_inches = "tight", transparent = True)
    
    df = process_text(df, col)

    create_wordcloud(df[df["sentiment"] == "Negative"][col].values, "negative")
    create_wordcloud(df[df["sentiment"] == "Positive"][col].values, "positive")

     
def clean_directory():
    """
    Cleans the root directory and static directory.
    """
    # delete any csv files from the root directory
    for file in os.listdir(THIS_FOLDER):
        if fnmatch.fnmatch(file, '*.csv'):
            file_path = os.path.join(THIS_FOLDER, file)
            print(file_path)
            os.remove(file_path)
    # delete any png files except the background file from the static directory
    for file in os.listdir(os.path.join(THIS_FOLDER, 'static')):
        if fnmatch.fnmatch(file, '*.png') and file != "background.png":
            img_path = os.path.join(THIS_FOLDER, f"static/{file}")
            print(img_path)
            os.remove(img_path)
            