import streamlit as st
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set up stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function for text preprocessing
def lemmatizing(content):
    content = re.sub(r'http\S+|www\S+|https\S+', '', content, flags=re.MULTILINE)
    content = re.sub(r'@\w+', '', content)
    content = re.sub('[^a-zA-Z]', ' ', content)

    content = content.lower()
    content = content.split()
    content = [lemmatizer.lemmatize(word) for word in content if word not in stop_words]
    return ' '.join(content)

# Load dataset for training the model
try:
    # Load Emotions.csv
    data = pd.read_csv("Emotions.csv")

    # Preprocess the data
    data['Reviews'] = data['Reviews'].apply(lemmatizing)

    # Split data into features and labels
    X = data['Reviews']
    y = data['Emotion']

    # Convert text data to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=2000, solver='liblinear')
    model.fit(X_train, y_train)

except FileNotFoundError:
    st.error("The file `Emotions.csv` is missing. Please ensure it is in the same directory as this script.")
    model = None
    vectorizer = None

# Streamlit App
st.title("Sentiment Analysis with Logistic Regression")
st.write("This app classifies reviews into emotion categories and provides sentiment summaries.")

# User Input Text Analysis
st.subheader("Test Your Text")
user_input = st.text_area("Enter text to classify:")
if st.button("Predict Emotion") and model and vectorizer:
    if user_input.strip():
        # Preprocess and vectorize the input
        processed_input = lemmatizing(user_input)
        input_vector = vectorizer.transform([processed_input])

        # Make prediction
        prediction = model.predict(input_vector)[0]

        # Categorize into Positive, Negative, Neutral
        positive_emotions = ['joy', 'love', 'admiration', 'gratitude', 'optimism', 'hope']
        negative_emotions = ['anger', 'sadness', 'fear', 'disgust', 'annoyance', 'grief']
        neutral_emotions = ['neutral', 'confusion']

        sentiment = (
            "Positive" if prediction in positive_emotions else
            "Negative" if prediction in negative_emotions else
            "Neutral"
        )

        # Display prediction and sentiment
        st.success(f"Predicted Emotion: {prediction}")
        st.info(f"Overall Sentiment: {sentiment}")
    else:
        st.error("Please enter valid text.")

# Upload and process CSV file
st.subheader("Bulk Analysis with CSV File")
uploaded_file = st.file_uploader("Upload a CSV file for bulk analysis", type=["csv"])

if uploaded_file is not None and model and vectorizer:
    try:
        # Read the uploaded CSV file
        uploaded_data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")

        # Let the user select the column containing reviews
        column_name = st.selectbox("Select the column containing reviews:", options=uploaded_data.columns)

        if column_name:
            # Preprocess the selected column
            uploaded_data['Processed_Reviews'] = uploaded_data[column_name].astype(str).apply(lemmatizing)

            # Perform predictions on the processed reviews
            predictions = model.predict(vectorizer.transform(uploaded_data['Processed_Reviews']))
            uploaded_data['Predicted_Emotion'] = predictions

            # Display categorized emotions
            st.subheader("Categorized Emotions")
            emotion_counts = uploaded_data['Predicted_Emotion'].value_counts()
            st.write(emotion_counts)

            # Categorize into Positive, Negative, Neutral
            uploaded_data['Sentiment'] = uploaded_data['Predicted_Emotion'].apply(
                lambda x: 'Positive' if x in positive_emotions else
                          ('Negative' if x in negative_emotions else 'Neutral')
            )

            sentiment_counts = uploaded_data['Sentiment'].value_counts()
            st.subheader("Sentiment Analysis Summary")
            st.write(sentiment_counts)

            # Allow user to download the results
            csv = uploaded_data.to_csv(index=False)
            st.download_button(label="Download Results", data=csv, file_name="analysis_results.csv", mime="text/csv")
        else:
            st.error("Please select a valid column containing review text.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

elif not model or not vectorizer:
    st.error("The model could not be loaded. Ensure the `Emotions.csv` file is present for training.")

else:
    st.info("Please upload a CSV file to proceed.")


