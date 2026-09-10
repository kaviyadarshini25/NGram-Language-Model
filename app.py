import streamlit as st
import re
import math
from collections import Counter


# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="N-Gram Language Model",
    page_icon="🧠",
    layout="wide"
)


# -----------------------------
# CUSTOM FRONTEND
# -----------------------------
st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 35px;
}

.section-title {
    font-size: 30px;
    font-weight: 650;
    margin-top: 35px;
    margin-bottom: 15px;
}

.result-card {
    padding: 15px;
    border-radius: 12px;
    background-color: #1f2937;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# TITLE
# -----------------------------
st.markdown(
    '<div class="main-title">🧠 N-Gram Language Model</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Next-Word Prediction using N-Grams and Laplace Smoothing'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# LOAD TRAINING CORPUS
# -----------------------------
try:

    with open("corpus.txt", "r", encoding="utf-8") as file:
        training_text = file.read()

except FileNotFoundError:

    training_text = """
    Natural language processing helps computers understand human language.
    Language models learn patterns from text.
    N gram models can predict the next word in a sentence.
    Unigram models consider one word at a time.
    Bigram models consider two words at a time.
    Trigram models consider three words at a time.
    Language models can predict words based on previous words.
    Artificial intelligence is changing the world.
    Machine learning helps computers learn from data.
    Natural language processing is an important field of artificial intelligence.
    Students can use language models for text prediction.
    Text prediction is used in search engines and mobile keyboards.
    Language models are useful for many applications.
    The model learns word patterns from training data.
    Laplace smoothing helps handle unseen word combinations.
    Smoothing prevents zero probability problems.
    Perplexity is used to evaluate language models.
    Lower perplexity generally indicates better prediction.
    """


# -----------------------------
# TOKENIZATION
# -----------------------------
def tokenize(text):

    text = text.lower()

    return re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )


# -----------------------------
# CREATE N-GRAMS
# -----------------------------
def create_ngrams(words, n):

    ngrams = []

    for i in range(len(words) - n + 1):

        ngram = tuple(
            words[i:i+n]
        )

        ngrams.append(ngram)

    return ngrams


# -----------------------------
# GET COUNTS
# -----------------------------
def get_counts(words, n):

    ngram_counts = Counter(
        create_ngrams(words, n)
    )

    if n == 1:

        context_counts = Counter()

    else:

        context_counts = Counter(
            create_ngrams(words, n - 1)
        )

    return ngram_counts, context_counts


# -----------------------------
# PROBABILITY
# -----------------------------
def calculate_probability(
    ngram,
    ngram_counts,
    context_counts,
    n,
    vocabulary_size,
    smoothing
):

    count = ngram_counts[ngram]

    # UNIGRAM
    if n == 1:

        total_words = sum(
            ngram_counts.values()
        )

        if smoothing:

            probability = (
                count + 1
            ) / (
                total_words + vocabulary_size
            )

        else:

            if total_words == 0:
                return 0

            probability = count / total_words

    # BIGRAM / TRIGRAM
    else:

        context = ngram[:-1]

        context_count = context_counts[
            context
        ]

        if smoothing:

            probability = (
                count + 1
            ) / (
                context_count + vocabulary_size
            )

        else:

            if context_count == 0:
                return 0

            probability = (
                count / context_count
            )

    return probability


# -----------------------------
# NEXT WORD PREDICTION
# -----------------------------
def predict_next_words(
    sentence,
    n,
    ngram_counts,
    context_counts,
    vocabulary,
    smoothing
):

    input_words = tokenize(sentence)

    if len(input_words) == 0:
        return []

    if n == 1:

        context = ()

    else:

        context = tuple(
            input_words[-(n-1):]
        )

    predictions = []

    for word in vocabulary:

        ngram = context + (word,)

        probability = calculate_probability(
            ngram,
            ngram_counts,
            context_counts,
            n,
            len(vocabulary),
            smoothing
        )

        if probability > 0:

            predictions.append(
                (word, probability)
            )

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return predictions[:10]


# -----------------------------
# PERPLEXITY
# -----------------------------
def calculate_perplexity(
    sentence,
    n,
    ngram_counts,
    context_counts,
    vocabulary,
    smoothing
):

    test_words = tokenize(sentence)

    if len(test_words) < n:
        return None

    probabilities = []

    for i in range(
        n - 1,
        len(test_words)
    ):

        ngram = tuple(
            test_words[i-n+1:i+1]
        )

        probability = calculate_probability(
            ngram,
            ngram_counts,
            context_counts,
            n,
            len(vocabulary),
            smoothing
        )

        if probability == 0:

            return float("inf")

        probabilities.append(
            probability
        )

    log_probability = 0

    for probability in probabilities:

        log_probability += math.log(
            probability
        )

    perplexity = math.exp(
        -log_probability /
        len(probabilities)
    )

    return perplexity


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.markdown(
    "## ⚙️ Model Settings"
)

n_choice = st.sidebar.selectbox(
    "Select N-Gram Model",
    [
        "Unigram",
        "Bigram",
        "Trigram"
    ]
)

if n_choice == "Unigram":

    n = 1

elif n_choice == "Bigram":

    n = 2

else:

    n = 3


smoothing = st.sidebar.checkbox(
    "Enable Laplace Smoothing",
    value=True
)


# -----------------------------
# TRAIN MODEL IN BACKGROUND
# -----------------------------
words = tokenize(training_text)

vocabulary = set(words)

ngram_counts, context_counts = get_counts(
    words,
    n
)


# ==================================================
# NEXT WORD PREDICTION
# ==================================================

st.markdown(
    '<div class="section-title">'
    '🔮 Next-Word Prediction'
    '</div>',
    unsafe_allow_html=True
)

sentence = st.text_input(
    "Enter a sentence:",
    "language"
)

if st.button(
    "✨ Predict Next Word",
    use_container_width=False
):

    predictions = predict_next_words(
        sentence,
        n,
        ngram_counts,
        context_counts,
        vocabulary,
        smoothing
    )

    if predictions:

        st.write("### Most Probable Next Words")

        for word, probability in predictions:

            st.markdown(
                f"""
                <div class="result-card">
                <b>{word}</b>
                &nbsp;&nbsp;→&nbsp;&nbsp;
                {probability:.4f}
                &nbsp;&nbsp;
                ({probability * 100:.2f}%)
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.warning(
            "No prediction available."
        )


# ==================================================
# GENERATED N-GRAMS
# ==================================================

st.markdown(
    '<div class="section-title">'
    '🔢 Generated N-Grams'
    '</div>',
    unsafe_allow_html=True
)

show_ngrams = st.checkbox(
    "Show N-Gram examples"
)

if show_ngrams:

    examples = list(
        ngram_counts.items()
    )[:20]

    for ngram, count in examples:

        text = " ".join(ngram)

        st.write(
            f"**{text}** → {count}"
        )


# ==================================================
# PERPLEXITY
# ==================================================

st.markdown(
    '<div class="section-title">'
    '📊 Perplexity Evaluation'
    '</div>',
    unsafe_allow_html=True
)

test_sentence = st.text_input(
    "Enter a test sentence:",
    "language models predict words"
)

if st.button(
    "📈 Calculate Perplexity"
):

    perplexity = calculate_perplexity(
        test_sentence,
        n,
        ngram_counts,
        context_counts,
        vocabulary,
        smoothing
    )

    if perplexity is None:

        st.warning(
            f"Enter at least {n} words "
            f"for a {n_choice} model."
        )

    elif math.isinf(perplexity):

        st.error(
            "Perplexity is infinite because "
            "an unseen word combination has "
            "zero probability."
        )

    else:

        st.success(
            f"Perplexity: {perplexity:.4f}"
        )

        st.info(
            "Lower perplexity generally "
            "indicates better prediction."
        )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "N-Gram Language Model • "
    "Next-Word Prediction • "
    "Laplace Smoothing • Perplexity"
)