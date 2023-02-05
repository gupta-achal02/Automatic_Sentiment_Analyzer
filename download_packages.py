import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

import gdown

url = "https://drive.google.com/file/d/1R4oKxOX75dTivjrBMMpdClBReVzpHoQa/view?usp=sharing"
output = "finetuned_bert_model"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)