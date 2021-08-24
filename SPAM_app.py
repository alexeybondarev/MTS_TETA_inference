from SPAM_classifier import SpamClassifier
import streamlit as st

st.title('SPAM SMS classification DEMO by fit-predict team')

st.markdown('Type or paste SMS message below')

with st.form('form'):
    text_input = st.text_area('Enter SMS message here: ', "I don't think this message is a SPAM")
    submit_button = st.form_submit_button('SPAM or not SPAM?')
    
    if submit_button:
        model = SpamClassifier()
        label = model.predict_text(text_input)
        if label == 0:
            st.write("That SMS message is not a SPAM")
            print("That is not a SPAM")
        elif label == 1:
            st.write("SPAM detected!")
        else:
            st.write("Oops, something went wrong")