from transformers import pipeline

# Create a sentiment-analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Example text for analysis
text = "I love using the Hugging Face transformers library!"

# Analyze sentiment
result = sentiment_pipeline(text)
print(result)
