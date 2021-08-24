from SPAM_classifier import SpamClassifier

text_input = "free entri in 005 text fa to 87121 to receiv entri"
model = SpamClassifier()
label = model.predict_text(text_input)
if label == 0:
    print("That is not a SPAM")
elif label == 1:
    print("SPAM detected!")
else:
    print("Oops, something went wrong")