import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle5
from sklearn.feature_extraction.text import CountVectorizer
import dill
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Reading all necessary pickle files

vocab = pickle5.load(open('count_vector.pickle', 'rb'))
Products_labels = pickle5.load(open('labels_dict_pandas.pickle', 'rb'))
tfidf_fit = pickle5.load(open('tfidf.pickle', 'rb'))
model = pickle5.load(open('lr_model.pickle', 'rb'))
clean_complaints_func = pickle5.load(open('dill_clean_func.pickle', 'rb'))
clean_data_function = dill.loads(clean_complaints_func)
app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    model_predict: str


@app.get('/')
def home():
    return 'Welcome to Automatic Ticket Classifier'


@app.post('/predict')
def predict_ticket(data: TextIn):
    cleaned_complaint_test1 = clean_data_function(data)
    count_vector_test1 = CountVectorizer(vocabulary=vocab)
    vectored_data = count_vector_test1.transform([cleaned_complaint_test1])
    tfidf_test1 = tfidf_fit.transform(vectored_data)
    predict_test1 = model.predict(tfidf_test1)
    model_predict = Products_labels[predict_test1[0]]
    print('Ticket belongs to the Product Category: ')
    return model_predict


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)


# uvicorn main:app --reload
