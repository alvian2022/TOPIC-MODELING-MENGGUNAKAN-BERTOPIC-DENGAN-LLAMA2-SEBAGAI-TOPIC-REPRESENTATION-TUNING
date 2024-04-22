import nltk
import re
import string
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
import pandas as pd
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemover, ArrayDictionary

data_tweet = pd.read_csv("tweetsdetikcom.csv", engine='python')

def filtering_text(text):
    # mengubah tweet menjadi huruf kecil
    text = text.lower()
    # menghilangkan url
    text = re.sub(r'https?:\/\/\S+','',text)
    # menghilangkan mention, link, hastag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    #menghilangkan karakter byte (b')
    text = re.sub(r'(b\'{1,2})',"", text)
    # menghilangkan yang bukan huruf
    text = re.sub('[^a-zA-Z]', ' ', text)
    # menghilangkan digit angka
    text = re.sub(r'\d+', '', text)
    #menghilangkan tanda baca
    text = text.translate(str.maketrans("","",string.punctuation))
    # menghilangkan whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#Proses stopwords dan stemming
def stop(text):
    #stopword
    with open('stopwords.txt') as kamus:
        word = kamus.readlines()
        list_stopword = [line.replace('\n',"") for line in word]
    dictionary = ArrayDictionary(list_stopword)
    stopword = StopWordRemover(dictionary)
    text = stopword.remove(text)
    return text

tweets = data_tweet['Tweet'].apply(filtering_text)
tweets = tweets.apply(stop)
tweets = tweets.to_list()

# print(tweets)

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine')
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', prediction_data=True)
ctfidf_model = ClassTfidfTransformer()

# Create the BERTopic model
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    ctfidf_model=ctfidf_model,
    language="indonesian",
    calculate_probabilities=True,
    verbose=True)

# Fit the BERTopic model to the preprocessed tweets data
topics, probs = topic_model.fit_transform(tweets)

topic_model.save("my_model")