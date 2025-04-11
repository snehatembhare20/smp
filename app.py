from flask import Flask, render_template, request
import pickle

# Load model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

vectorizer = model_data['vectorizer']
model = model_data['model']

# Flask setup
app = Flask(__name__)

# Emoji map
emoji_map = {
    "positive": ("😊", "Happy"),
    "negative": ("😠", "Angry")
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    emoji = ""
    emotion = ""
    
    if request.method == 'POST':
        text = request.form['review']
        
        # Simple clean-up like earlier
        import re
        import string
        from nltk.corpus import stopwords
        import nltk
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            text = text.lower()
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"[^a-zA-Z]", " ", text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = text.split()
            tokens = [word for word in tokens if word not in stop_words]
            return ' '.join(tokens)
        
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)[0]
        
        emoji, emotion = emoji_map[result]
        prediction = result.capitalize()

    return render_template("index.html", prediction=prediction, emoji=emoji, emotion=emotion)

if __name__ == '__main__':
    app.run(debug=True, port=5000)