import pandas as pd
import nltk
nltk.download('vader_lexicon')
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from scipy.special import softmax

plt.style.use('ggplot')

# Load dataset
file_path = 'synthetic_reviews.csv'

try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully with shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
    raise

if df is not None:
    print(df.head())
else:
    print("DataFrame is empty. Please ensure the file path is correct.")

# VADER sentiment analysis
sia = SentimentIntensityAnalyzer()

example = df['Text'][50]
print(f"Example review text: {example}")

vader_scores = sia.polarity_scores(example)
print(f"VADER Sentiment Scores: {vader_scores}")

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    res[i] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Row_Index'})
vaders = vaders.merge(df, how='left', left_on='Row_Index', right_index=True)

# Plot VADER results
# ax = sns.barplot(data=vaders, x='Sentiment', y='compound')
# ax.set_title('Compound Score by Sentiment')
# plt.show()

# fig, axs = plt.subplots(1, 3, figsize=(12, 3))
# sns.barplot(data=vaders, x='Sentiment', y='pos', ax=axs[0])
# sns.barplot(data=vaders, x='Sentiment', y='neu', ax=axs[1])
# sns.barplot(data=vaders, x='Sentiment', y='neg', ax=axs[2])
# axs[0].set_title('Positive')
# axs[1].set_title('Neutral')
# axs[2].set_title('Negative')
# plt.tight_layout()
# plt.show()

# Load Hugging Face model using TensorFlow
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

# Function to calculate Roberta sentiment scores
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='tf')  # Use TensorFlow tensors
    output = model(encoded_text)
    scores = output.logits.numpy()[0]  # Get logits and convert to NumPy
    scores = softmax(scores)  # Apply softmax to logits
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

# Perform sentiment analysis for all rows
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {f"vader_{key}": value for key, value in vader_result.items()}
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[i] = both
    except RuntimeError:
        print(f'Broke for row {i}')

# Create results DataFrame
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Row_Index'})
results_df = results_df.merge(df, how='left', left_on='Row_Index', right_index=True)

# Plot results
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue='Sentiment', palette='tab10')

positive_1_star_review = results_df.query('Sentiment == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]
negative_5_star_review = results_df.query('Sentiment == 0') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0]

print("Positive Review Example: ", positive_1_star_review)
print("Negative Review Example: ", negative_5_star_review)

# # Use Hugging Face pipeline for quick sentiment analysis
# from transformers import pipeline
# sent_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# print(sent_pipeline('I love sentiment analysis!'))
# print(sent_pipeline('Make sure to like and subscribe!'))
