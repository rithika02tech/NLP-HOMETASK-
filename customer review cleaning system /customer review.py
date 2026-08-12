import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

df = pd.read_csv("data/IMDB Dataset.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Reviews:")
print(df['review'].duplicated().sum())


df = df.dropna()
df = df.drop_duplicates(subset='review')


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def clean_review(text):

    text = text.lower()

   
    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'http\S+|www\S+', '', text)

   
    text = re.sub(r'\d+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)


    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

df['cleaned_review'] = df['review'].apply(clean_review)

print("\nOriginal vs Cleaned Reviews:")
print(df[['review', 'cleaned_review']].head(10))
df.to_csv("preprocessed_imdb.csv", index=False)

print("\nPreprocessed dataset exported successfully!")
print("File: preprocessed_imdb.csv")
