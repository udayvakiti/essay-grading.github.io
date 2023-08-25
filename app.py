# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:43:17 2023

@author: udaykiranreddyvakiti
"""

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

app = Flask(__name__)

# Load the pre-trained LSTM model and tokenizer
model = load_model('essay_grading_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/grade')
def grade():
    essay = request.form['essay']

    # Tokenize the essay
    essay_sequence = tokenizer.texts_to_sequences([essay])

    # Pad the sequence
    max_length = 500
    essay_data = pad_sequences(essay_sequence, maxlen=max_length)

    # Make the prediction
    predicted_score = model.predict(essay_data)[0][0]

    # Round the score to two decimal places
    predicted_score = round(predicted_score, 2)

    return render_template('index.html', grade=predicted_score)

if __name__ == '__main__':
    app.run(debug=True)
