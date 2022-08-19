import os
import fnmatch
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, send_file
from flask.templating import render_template
from flask_session import Session
from functions import *

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
print("APP")
print(THIS_FOLDER)

@app.route('/', methods=["GET", "POST"])
def index():
    # User reached route via POST
    if request.method == "POST":

        clean_directory()
        
        f = request.files["file"]
        file_path = os.path.join(THIS_FOLDER, secure_filename(f.filename))
        f.save(file_path)

        # getting the value for 'column'
        column = request.form.get("column")
        
        # creating dataframe from uploaded file
        df = get_df(file_path[:-4], column)

        # assigning sentiments
        df = add_sentiments(df, column)

        # getting output csv
        get_output(df, file_path[:-4])

        # getting plots
        get_plots(df, column)

        # render 'result.html'
        return render_template("result.html")

    # render 'index.html'
    return render_template("index.html")


@app.route('/download')
def download():
    for file in os.listdir(THIS_FOLDER):
        if fnmatch.fnmatch(file, '*_output.csv'):
            file_path = os.path.join(THIS_FOLDER, file)
    return send_file(file_path, as_attachment = True)


if __name__ == '__main__':
    app.run(threaded=True, port=int(os.environ.get("PORT", 5000)))
