# N-Gram Language Model with Laplace Smoothing

## Project Description

This mini project implements an N-Gram Language Model for next-word prediction.

The model supports:

- Unigram
- Bigram
- Trigram
- Laplace Smoothing
- Next-word Prediction
- Perplexity Evaluation

## Technologies Used

- Python
- Streamlit

## How to Run

Open the project folder in VS Code.

Install the required package:

pip install -r requirements.txt

Run the application:

streamlit run app.py

The application will open in the browser.

## Working

The model learns word patterns from the training corpus.

Based on the selected N-Gram model, it calculates word probabilities
and predicts the most probable next words.

Laplace smoothing can be enabled to avoid zero probability for
unseen word combinations.

Perplexity is used to evaluate the performance of the language model.