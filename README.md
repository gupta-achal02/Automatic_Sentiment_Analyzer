
# Automatic Sentiment Analyzer

ASA is an app that takes a complete csv file from the user as an input and assigns a sentiment label to each row. It then returns an output csv file with a new column for the sentiment labels.

It uses a [DistilBERT](https://arxiv.org/abs/1910.01108) model that [I fintuned on the IMDB dataset](https://github.com/hailASG/Finetuned_DistilBERT) and the [VADER](https://ojs.aaai.org/index.php/icwsm/article/view/14550)  to sentiemnt analyzer assing the sentiment labels. 

What to choose from these models depends on the number of rows in the csv file, as using DistilBERT with larger files will significantly increase the runtime.

[Demo](http://automaticanalyzer.pythonanywhere.com/)

## Setup
- You'll first need to install [PyTorch](https://pytorch.org/). It will be needed for using the DistilBERT model.
- Then install the requirements using `pip install -r reqirements.txt`
- Finally run the python file `download_packages.py`. It will download the needed NLTK packages and the DistilBERT model from my Google Drive.

## Working
You can start the app by running `app.py` file.

On the first page, you must upload the csv file and provide the column name for which you want sentiments to be analysed, and then click on submit.

![Page 1](https://github.com/hailASG/Automatic_Sentiment_Analyzer/blob/main/images/1.png)

These are the last few columns of the input csv file that I've chosen in this example

![Page 6](https://github.com/hailASG/Automatic_Sentiment_Analyzer/blob/main/images/6.png)

After some time a new page will open. Here you can press the Download button to download the output csv file. It will be named `{input_filename}_output.csv`. If you scroll down, you can see a countplot, piechart and two wordclouds for the negative and positive labels.

![Page 2](https://github.com/hailASG/Automatic_Sentiment_Analyzer/blob/main/images/2.png)![Page 3](https://github.com/hailASG/Automatic_Sentiment_Analyzer/blob/main/images/3.png)![Page 4](https://github.com/hailASG/Automatic_Sentiment_Analyzer/blob/main/images/4.png)![Page 5](https://github.com/hailASG/Automatic_Sentiment_Analyzer/blob/main/images/5.png)

The resultant csv file now has an additional column with the name 'sentiment'

![Page 7](https://github.com/hailASG/Automatic_Sentiment_Analyzer/blob/main/images/7.png)

## Files
There are two main files `app.py` and `functions.py`. 

`app.py` file contains the code for the Flask app itself, while `functions.py` file contains all the required functions for loading the DistilBERT model and the VADER sentiment analyzer, reading the csv file, and assinging the sentiment labels to create the output csv file.
