import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import fnmatch
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
lemma = WordNetLemmatizer()
stop_words = stopwords.words("english")

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
print("Functions")
print(THIS_FOLDER)

def get_df(file_path, col):
    df = pd.read_csv(f"{file_path}.csv")
    df.rename(columns = lambda x: x.strip(), inplace = True)
    # Dropping duplicated rows
    df = df.drop_duplicates(subset = col, keep = "first")
    # Drop rows with missing values
    df = df.dropna(subset = [col])
    # Resetting the index of the dataframe
    df = df.reset_index(drop = True)
    return df


def set_sentiment(text):
    polarity_dict = analyser.polarity_scores(text)
    if polarity_dict['compound'] >= 0.05 :
        return "Positive"
    elif polarity_dict['compound'] <= - 0.05 :
        return "Negative"
    else :
        return "Neutral"


def add_sentiments(df, col):
    df["sentiment"] = df[col].apply(lambda x: set_sentiment(x))
    return df


def get_output(df, file_path):
    df.to_csv(f"{file_path}_output.csv", index = False)


def tokenize_text(df, col):
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
    words = ' '.join([words for words in text])
    wordcloud = WordCloud(width = 800, height = 400, max_font_size = 70, 
                          max_words = 100, background_color = "white").generate(words)
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation = "bilinear")
    plt.axis("off")
    plt.savefig(f"static/wordcloud_{sentiment}.png", bbox_inches = "tight", transparent = True)


def get_plots(df, col):

    cp = sns.countplot(x = df["sentiment"])
    fig1 = cp.get_figure()
    fig1.savefig("static/countplot.png", bbox_inches = "tight", transparent = True)
    plt.close(fig1)
    
    fig2 = plt.gcf()
    fig2.set_size_inches(7,7)

    colors = ["#66b3ff", "#99ff99", "#ff9999"]
    plt.pie(df["sentiment"].value_counts(), labels = ("Positive", "Neutral", "Negative"), radius = 3, autopct = "%1.1f%%",
            shadow = True, startangle = 90, labeldistance = 1.1, colors = colors, explode = (0.1, 0.1, 0.1))
    plt.axis("equal")
    plt.savefig("static/pie_chart.png", bbox_inches = "tight", transparent = True)
    
    df = tokenize_text(df, col)

    create_wordcloud(df[df["sentiment"] == "Negative"][col].values, "negative")
    create_wordcloud(df[df["sentiment"] == "Positive"][col].values, "positive")

     
def clean_directory():
    for file in os.listdir(THIS_FOLDER):
        if fnmatch.fnmatch(file, '*.csv'):
            file_path = os.path.join(THIS_FOLDER, file)
            print(file_path)
            os.remove(file_path)
    
    for file in os.listdir(os.path.join(THIS_FOLDER, 'static')):
        if fnmatch.fnmatch(file, '*.png') and file != "background.png":
            img_path = os.path.join(THIS_FOLDER, f"static/{file}")
            print(img_path)
            os.remove(img_path)

    
    




