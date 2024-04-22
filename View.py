import streamlit as st
from Pemodelan_Topik import *

class View:
    def __init__(self):
        self.Pemodelan_Topik = Pemodelan_Topik()
    
    def run(self):
        uploaded_file = self.form()

        if uploaded_file is not None:
            self.Pemodelan_Topik.load_data(uploaded_file)
            self.Pemodelan_Topik.preprocess_tweets()
            self.display_preprocessed_tweets()

            sample_form = st.form("Form Data Tweet")
            bt = sample_form.form_submit_button("Extract Topic")

            if bt:
                self.Pemodelan_Topik.load_pemodelan_topik()
                self.Pemodelan_Topik.transform_pemodelan_topik()
                self.Pemodelan_Topik.evaluate_pemodelan_topik()
                self.display_pemodelan_topic_result()
                self.display_evaluate_pemodelan_topik()
                self.display_visualize_topics()
                self.display_visualize_pemodelan_topik()
    
    def form(self):
        st.title("TOPIC MODELING MENGGUNAKAN BERTOPIC DENGAN LLAMA2 SEBAGAI TOPIC REPRESENTATION TUNING")

        # Add a file uploader widget
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        return uploaded_file

    def display_preprocessed_tweets(self):
        st.header("Dataset")
        st.dataframe(pd.DataFrame(self.Pemodelan_Topik.tweets, columns=["Tweets"]), width=720)
    
    def display_pemodelan_topic_result(self):
        # Get topic info
        # topic_info = self.Pemodelan_Topik.BERTopic_model.get_topic_info()
        topic_info = self.Pemodelan_Topik.topic_results_testing
        st.subheader("Daftar Topik")
        st.dataframe(topic_info, width=720)
        st.write('Jumlah Topik ', self.Pemodelan_Topik.jml_topik)
    
    def display_visualize_pemodelan_topik(self):
        st.subheader("Topic Similarity Matrix")
        fig = self.Pemodelan_Topik.BERTopic_model.visualize_heatmap()
        st.write(fig)

    def display_visualize_topics(self):
        st.subheader("Topics")
        fig = self.Pemodelan_Topik.BERTopic_model.visualize_topics()
        st.write(fig)

    def display_evaluate_pemodelan_topik(self):
        st.subheader("Coherence Score Setiap Topik")
        st.dataframe(self.Pemodelan_Topik.coherence_score_topics, width=720)
        st.write('Rata-rata Coherence Score', self.Pemodelan_Topik.average_coherence_score)
        st.write('Coherence Score Tertinggi', self.Pemodelan_Topik.max_coherence_score)
        st.write('Topik ', self.Pemodelan_Topik.max_coherence_topic_id)
        st.write('Daftar Kata : ', self.Pemodelan_Topik.max_coherence_daftar_kata)
        st.write('Coherence Score Terendah', self.Pemodelan_Topik.min_coherence_score)
        st.write('Topik ', self.Pemodelan_Topik.min_coherence_topic_id)
        st.write('Daftar Kata : ', self.Pemodelan_Topik.min_coherence_daftar_kata)

    
tweet_topic_modeling = View()
tweet_topic_modeling.run()

