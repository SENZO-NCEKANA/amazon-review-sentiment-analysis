"""
Sentiment Analysis for Amazon Product Reviews.

This script loads Amazon product reviews, preprocesses the text,
and performs sentiment analysis using spaCy and SpacyTextBlob.
"""

import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob


def load_spacy_model():
    """
    Load the spaCy language model and add the SpacyTextBlob pipeline.

    Returns:
        spacy.Language: Loaded spaCy NLP model.
    """
    nlp = spacy.load("en_core_web_md")

    if "spacytextblob" not in nlp.pipe_names:
        nlp.add_pipe("spacytextblob")
        nlp.meta["spacytextblob_component"] = SpacyTextBlob.__name__

    return nlp


def preprocess_text(text, nlp_model):
    """
    Clean review text by removing stop words, punctuation,
    and converting words to lowercase lemmas.

    Args:
        text (str): Original review text.
        nlp_model (spacy.Language): spaCy NLP model.

    Returns:
        str: Cleaned review text.
    """
    doc = nlp_model(text.lower().strip())

    clean_tokens = [
        token.lemma_.lower().strip()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.text.strip()
    ]

    return " ".join(clean_tokens)


def analyze_sentiment(review_text, nlp_model):
    """
    Predict the sentiment of a review using SpacyTextBlob.

    Args:
        review_text (str): Preprocessed review text.
        nlp_model (spacy.Language): spaCy NLP model.

    Returns:
        tuple:
            Polarity score and sentiment label
            (Positive, Negative, Neutral).
    """
    doc = nlp_model(review_text)
    polarity = doc._.blob.polarity

    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return polarity, sentiment


def main():
    """
    Load the dataset, perform sentiment analysis on sample reviews,
    and display similarity between two reviews.
    """
    nlp_model = load_spacy_model()

    file_path = (
        "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
    )

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return

    # Drop rows with missing review text
    clean_df = df.dropna(subset=["reviews.text"])
    reviews_data = clean_df["reviews.text"]

    print("\n--- Sentiment Analysis Samples ---")
    sample_indices = [0, 50, 100]

    for idx in sample_indices:
        # Safety check to avoid out-of-range errors
        if idx >= len(reviews_data):
            continue

        original_review = reviews_data.iloc[idx]
        processed_review = preprocess_text(original_review, nlp_model)
        polarity, sentiment = analyze_sentiment(
            processed_review,
            nlp_model,
        )

        print(f"\nReview Index {idx}:")
        print(f"Original: {original_review[:100]}...")
        print(f"Sentiment: {sentiment} (Polarity: {polarity:.2f})")

    print("\n--- Similarity Testing ---")
    # Example similarity between first two reviews
    if len(reviews_data) >= 2:
        review_a = nlp_model(
            preprocess_text(reviews_data.iloc[0], nlp_model)
        )
        review_b = nlp_model(
            preprocess_text(reviews_data.iloc[1], nlp_model)
        )
        similarity_score = review_a.similarity(review_b)
        print(
            "Similarity score between Review 0 and Review 1: "
            f"{similarity_score:.2f}"
        )
    else:
        print("Not enough reviews to calculate similarity.")


if __name__ == "__main__":
    main()
