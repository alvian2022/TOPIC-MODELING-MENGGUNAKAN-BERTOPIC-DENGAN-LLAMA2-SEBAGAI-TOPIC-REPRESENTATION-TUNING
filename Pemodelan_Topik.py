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
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns

class Pemodelan_Topik:
    def __init__(self):
        self.data = None
        self.tweets = None
        self.BERTopic_model = None
        self.jml_topik = None
        self.average_coherence_score = None
        self.coherence_score_topics = None
        self.min_coherence_score = None
        self.min_coherence_topic_id = None
        self.min_coherence_daftar_kata = None
        self.max_coherence_score = None
        self.max_coherence_topic_id = None
        self.max_coherence_daftar_kata = None
        self.topic_results = None
        self.topic_results_testing = None
      

    def load_data(self, uploaded_file):
        # Use pandas to read the uploaded file
        self.data = pd.read_csv(uploaded_file, engine='python')

    def preprocess_text(self, text):
        # mengubah tweet menjadi huruf kecil
        # text = text.lower()
        # menghilangkan url
        text = re.sub(r'https?:\/\/\S+','',text)
        # menghilangkan mention, link, hastag
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
        #menghilangkan karakter byte (b')
        text = re.sub(r'(b\'{1,2})',"", text)
        # menghilangkan yang bukan huruf
        # text = re.sub('[^a-zA-Z]', ' ', text)
        # # menghilangkan digit angka
        # text = re.sub(r'\d+', '', text)
        # #menghilangkan tanda baca
        # text = text.translate(str.maketrans("","",string.punctuation))
        # # menghilangkan whitespace berlebih
        # text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_tweets(self):
        self.tweets = self.data['Tweet'].apply(self.preprocess_text).to_list()

    def load_pemodelan_topik(self):
        self.BERTopic_model = BERTopic.load("/content/drive/MyDrive/Skripsi/Bertopic_Llama2/final_5_training",  embedding_model='firqaaa/indo-sentence-bert-base')

    def transform_pemodelan_topik(self):
        # Fit the model
        topics, probs = self.BERTopic_model.transform(self.tweets)
        self.data['topic'] = pd.Series(topics)
        return topics

    def count_words(self, id_topic):
        list_sentence = self.data[self.data['topic'] == id_topic]['Tweet'].to_list()
        sentence = ''.join(list_sentence)
        words = sentence.split()
        num_words = len(words)
        return num_words


    def evaluate_pemodelan_topik(self):
        topics = self.transform_pemodelan_topik()

        documents = pd.DataFrame({"Document": self.tweets,
                                  "ID": range(len(self.tweets)),
                                  "Topic": topics})

        # documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        documents_per_topic = documents.groupby('Topic', as_index=False).agg(Document=('Document', ' '.join))
        documents_per_topic['Document_Count'] = documents.groupby('Topic').size().reset_index(name='Document_Count')['Document_Count']
        
        cleaned_docs = self.BERTopic_model._preprocess_text(documents_per_topic.Document.values)
        # self.tweets = self.BERTopic_model._preprocess_text(self.tweets)
        # Extract vectorizer and tokenizer from BERTopic
        vectorizer = self.BERTopic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()
        # tokenizer = vectorizer.build_tokenizer()

        # Extract features for Topic Coherence evaluation
        words = vectorizer.get_feature_names_out()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        # BERTopic_model = self.BERTopic_model

        # Extract words in each topic if they are non-empty and exist in the dictionary
        duplicate_BERTopic_model = self.BERTopic_model
        topic_words = []
        for topic in range(-1, len(set(topics))-1):
            words = list(zip(*duplicate_BERTopic_model.get_topic(topic)))[0]
            words = [word for word in words if word in dictionary.token2id]
            # words = [word for word in words if isinstance(word, str) and word in dictionary.token2id]
            topic_words.append(words)
        topic_words = [words for words in topic_words if len(words) > 0]

        # Evaluate
        coherence_model_cv = CoherenceModel(topics=topic_words,
                                texts=tokens,
                                corpus=corpus,
                                dictionary=dictionary,
                                coherence='c_v')

        self.average_coherence_score = coherence_model_cv.get_coherence()

        topic_coherence_cv = coherence_model_cv.get_coherence_per_topic(segmented_topics=None, with_std=False, with_support=False)

        topic_words = [[words for words, _ in self.BERTopic_model.get_topic(topic)] 
              for topic in range(-1, len(set(topics))-1)]

        x = len(topic_words)
        y = len(topic_words[0])
        z = len(topic_words[0][0])

        daftar_topik = []
        daftar_kata_arr = []
        for i in range(x):
            daftar_kata = ""
            for j in range(y):
                o = topic_words[i][j]
                daftar_kata_arr.append(o)
            daftar_kata = ", ".join(daftar_kata_arr)
            daftar_topik.append(daftar_kata)
            daftar_kata_arr.clear()

        # topic_id = []
        # topic_coherence = []
        # # topic_words = []

        # i = -1
        # for topics_coherence in topic_coherence_cv:
        #     topic_id.append(i)
        #     topic_coherence.append(topics_coherence)
        #     i += 1

        self.coherence_score_topics = pd.DataFrame(
            {
                "Topic":  range(-1, len(daftar_topik)-1),
                "Daftar_Kata": daftar_topik,
                "Cohrence Score CV": topic_coherence_cv,
            }
        )
          
        jumlah_kata = [len(i) for i in tokens]
        topic_representation = []

        for i in  documents_per_topic['Topic']:
          topic_i = self.BERTopic_model.get_topic(i, full=True)['Llama2'][0][0]
          topic_representation.append(topic_i)
        # for i in self.coherence_score_topics['Topic']:
        #   num_words.append(self.count_words(i))

        self.topic_results_testing = pd.DataFrame(
              {
                  "Topic":  documents_per_topic['Topic'],
                  "Jumlah Tweet": documents_per_topic['Document_Count'],
                  "Jumlah Kata": jumlah_kata,
                  "Representasi Topik": topic_representation,
              }
          )




        # self.coherence_score_topics['Jumlah Kata'] = pd.Series(num_words)
        id_min = topic_coherence_cv.index(min(topic_coherence_cv))
        id_max = topic_coherence_cv.index(max(topic_coherence_cv))
        self.jml_topik = self.coherence_score_topics['Topic'].nunique()

        self.min_coherence_topic_id = self.coherence_score_topics[self.coherence_score_topics['Cohrence Score CV'] == self.coherence_score_topics['Cohrence Score CV'].min()]['Topic'].values[0]
        self.min_coherence_score = self.coherence_score_topics['Cohrence Score CV'].min()
        self.min_coherence_daftar_kata = self.coherence_score_topics[self.coherence_score_topics['Cohrence Score CV'] == self.coherence_score_topics['Cohrence Score CV'].min()]['Daftar_Kata'].values[0]
        self.max_coherence_topic_id = self.coherence_score_topics[self.coherence_score_topics['Cohrence Score CV'] == self.coherence_score_topics['Cohrence Score CV'].max()]['Topic'].values[0]
        self.max_coherence_score = self.coherence_score_topics['Cohrence Score CV'].max()
        self.max_coherence_daftar_kata = self.coherence_score_topics[self.coherence_score_topics['Cohrence Score CV'] == self.coherence_score_topics['Cohrence Score CV'].max()]['Daftar_Kata'].values[0]
