import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = TFAutoModelForSequenceClassification.from_pretrained("VinayakMane47/bert-base-cased-finetuned-on-duplicate-Q-A")
import streamlit as st

# Importing the required function
def check_similarity(question1, question2, debug=0):
    tokenizer_output = tokenizer(question1, question2, truncation=True, return_token_type_ids=True, max_length=75, return_tensors='tf')
    logits = model(**tokenizer_output)["logits"]
    predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
    if predicted_class_id == 1:
        if debug:
            print("Meaning of Both the Questions is same")
        return 1
    else:
        if debug:
            print("Both Questions are different")
        return 0

# Setting up the Streamlit app
st.title("Question Similarity Checker")

# Text input boxes for the two questions
question1 = st.text_input("Enter question 1")
question2 = st.text_input("Enter question 2")

# Button to trigger similarity check
if st.button("Check Similarity"):
    # Checking the similarity and displaying the result
    similarity_score = check_similarity(question1, question2)
    if similarity_score == 1:
        st.write("Meaning of Both the Questions is same")
    else:
        st.write("Both Questions are different")
