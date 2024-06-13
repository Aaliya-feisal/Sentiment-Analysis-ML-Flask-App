from flask import Flask, request, render_template
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Route for splash page
@app.route('/')
def index():
    return render_template('index.html')

# Route for text input page
@app.route('/input_text')
def input_text():
    return render_template('input_text.html')

# Route for URL input page
@app.route('/input_url')
def input_url():
    return render_template('input_url.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.form['text_input'].lower()
    text_final = ''.join(c for c in text if not c.isdigit())
    processed_text = ' '.join([word for word in text_final.split() if word not in stop_words])

    # Use specific model for sentiment analysis
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_pipeline(processed_text)[0]
    
    sentiment = result['label'].lower()
    score = round(result['score'], 2)

    return render_template('input_text.html', 
                           final=score, 
                           text1=processed_text,
                           text2=result['score'],
                           text5=result['score'],
                           text4=score,
                           text3=result['score'],
                           sentiment=sentiment,
                           original_text=text)

# Route for processing URL input
@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    url = request.form['url_input']
    reviews = scrape_reviews(url)
    processed_text = ' '.join(reviews)
    
    # Use specific model for sentiment analysis
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_pipeline(processed_text)[0]
    
    sentiment = result['label'].lower()
    score = round(result['score'], 2)

    return render_template('input_url.html', 
                           final=score, 
                           text1=processed_text,
                           text2=result['score'],
                           text5=result['score'],
                           text4=score,
                           text3=result['score'],
                           sentiment=sentiment,
                           original_text=url)


def scrape_reviews(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    reviews = []
    for review in soup.find_all('span', {'data-hook': 'review-body'}):
        reviews.append(review.get_text().strip().lower())
    return reviews

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
