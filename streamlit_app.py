import streamlit as st
import time
import re
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA, TruncatedSVD
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

placeholderstr = "Please input your command"
user_name = "Lydia"
user_image = "https://www.w3schools.com/howto/img_avatar.png"

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

def preprocess_sentences(sentences):
    return [simple_preprocess(s) for s in sentences]

def train_word2vec(sentences_tokens, sg=1, window=5, vector_size=100, epochs=50, min_count=1):
    model = Word2Vec(sentences_tokens, sg=sg, window=window, vector_size=vector_size, min_count=min_count, epochs=epochs)
    return model

def word_vectors(model):
    words = []
    vecs = []
    for word in model.wv.index_to_key:
        words.append(word)
        vecs.append(model.wv[word])
    return np.array(vecs), words

def plot_2d(vectors, labels, title):
    df = np.vstack((vectors[:,0], vectors[:,1])).T
    fig = px.scatter(x=df[:,0], y=df[:,1], text=labels, title=title, width=700, height=500)
    fig.update_traces(textfont_size=12, textposition='top center')
    st.plotly_chart(fig)

def plot_3d(vectors, labels, title):
    df = np.vstack((vectors[:,0], vectors[:,1], vectors[:,2])).T
    fig = px.scatter_3d(x=df[:,0], y=df[:,1], z=df[:,2], text=labels, title=title, width=700, height=600)
    fig.update_traces(textfont_size=10)
    st.plotly_chart(fig)

def generate_response(prompt):
    pattern = r'\b(i(\'?m| am| feel| think i(\'?)?m)?\s*(so\s+)?(stupid|ugly|dumb|idiot|worthless|loser|useless))\b'
    if re.search(pattern, prompt, re.IGNORECASE):
        return "Yes, you are!"
    else:
        return f"You say: {prompt}."

def main():
    st.set_page_config(page_title='K-Assistant - The Residemy Agent', layout='wide', initial_sidebar_state='auto')
    pages = ["Chat", "Q1-1: 2D & 3D", "Q2: Skip-gram", "Q3: CBOW"]
    selected_page = st.sidebar.radio("Go to", pages)
    st.sidebar.image(user_image)
    st.sidebar.write(f"User: {user_name}")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sentences" not in st.session_state:
        st.session_state.sentences = []

    if selected_page == "Chat":
        st.title(f"\U0001F4AC {user_name}'s Chatbot")
        st_c_chat = st.container()

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st_c_chat.chat_message("user", avatar=user_image).markdown(msg["content"])
            else:
                st_c_chat.chat_message("assistant").markdown(msg["content"])

        if prompt := st.chat_input(placeholder=placeholderstr):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.sentences.append(prompt)
            st_c_chat.chat_message("user", avatar=user_image).write(prompt)
            response = generate_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st_c_chat.chat_message("assistant").write_stream(stream_data(response))
        return

    sentences = st.session_state.sentences
    tokens = preprocess_sentences(sentences) if sentences else []

    if selected_page == "Q1-1: 2D & 3D":
        st.title("Question 1-1: 2D & 3D View")
        if sentences:
            model_w2v = train_word2vec(tokens, sg=1, window=5, vector_size=50, epochs=100)
            vecs, words = word_vectors(model_w2v)
            st.subheader("Word2Vec PCA")
            pca = PCA(n_components=3, random_state=42)
            pca_vecs = pca.fit_transform(vecs)
            plot_2d(pca_vecs, words, "PCA (2D)")
            plot_3d(pca_vecs, words, "PCA (3D)")
            st.subheader("Word2Vec SVD")
            svd = TruncatedSVD(n_components=3, random_state=42)
            svd_vecs = svd.fit_transform(vecs)
            plot_2d(svd_vecs, words, "SVD (2D)")
            plot_3d(svd_vecs, words, "SVD (3D)")
        else:
            st.info("sentence")

    if selected_page == "Q2: Skip-gram":
        st.title("Question 2: Skip-gram Parameter Tuning")
        if sentences:
            window = st.sidebar.slider("Window size", 1, 10, 5)
            vector_size = st.sidebar.slider("Vector size", 10, 200, 50)
            epochs = st.sidebar.slider("Epochs", 10, 200, 100)
            model_sg = train_word2vec(tokens, sg=1, window=window, vector_size=vector_size, epochs=epochs)
            st.success("Skip-gram model trained.")
            test_word = st.text_input("Enter a word to test (must exist in vocabulary):")
            if test_word:
                if test_word in model_sg.wv:
                    similar_words = model_sg.wv.most_similar(test_word, topn=5)
                    st.write(f"Top 5 similar words to '{test_word}':")
                    for word, score in similar_words:
                        st.write(f"{word} (Score: {score:.4f})")
                else:
                    st.error(f"'{test_word}' not found in vocabulary.")
        else:
            st.info("sentence...")

    if selected_page == "Q3: CBOW":
        st.title("Question 3: CBOW Parameter Tuning")
        if sentences:
            window = st.sidebar.slider("Window size", 1, 10, 5, key="cbow_win")
            vector_size = st.sidebar.slider("Vector size", 10, 200, 50, key="cbow_vec")
            epochs = st.sidebar.slider("Epochs", 10, 200, 100, key="cbow_ep")
            model_cbow = train_word2vec(tokens, sg=0, window=window, vector_size=vector_size, epochs=epochs)
            st.success("CBOW model trained.")
            test_word = st.text_input("Enter a word to test (must exist in vocabulary):", key="cbow_test")
            if test_word:
                if test_word in model_cbow.wv:
                    similar_words = model_cbow.wv.most_similar(test_word, topn=5)
                    st.write(f"Top 5 similar words to '{test_word}':")
                    for word, score in similar_words:
                        st.write(f"{word} (Score: {score:.4f})")
                else:
                    st.error(f"'{test_word}' not found in vocabulary.")
        else:
            st.info("sentence...")

if __name__ == "__main__":
    main()