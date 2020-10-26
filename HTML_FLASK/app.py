import io
import base64
import matplotlib
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from jinja2 import Template
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from flask import Flask, render_template, request
from sklearn.cluster import KMeans, MiniBatchKMeans
from sentence_transformers import SentenceTransformer, util
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# KorSBERT Trained Model
model_path = './output/uvvu16seedgojung_con_last'

# Embedding objective
embedder = SentenceTransformer(model_path)

# Corpus with example sentences for Semantic Search (fix default)
corpus = ['한 남자가 음식을 먹는다.',
          '한 남자가 빵 한 조각을 먹는다.',
          '그 여자가 아이를 돌본다.',
          '한 남자가 말을 탄다.',
          '한 여자가 바이올린을 연주한다.',
          '두 남자가 수레를 숲 속으로 밀었다.',
          '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
          '원숭이 한 마리가 드럼을 연주한다.',
          '치타 한 마리가 먹이 뒤에서 달리고 있다.']

# Represent Embedding corpus
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Flask app
app = Flask(__name__)

# Index html
@app.route('/')
def OpenMainPage():
    return render_template('main_page.html')

# Semantic Search html
@app.route('/semantic_search', methods=['GET'])
def SemanticeSearch():
    # Top k most similar senteces
    top_k = 5

    # Get user query sentence (ex : 한 남자가 파스타를 먹는다.)
    query = request.args.get('query')

    # Represent Embedding query
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # Calculate cosin similarity between query and corpus
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    # Use np.argpartition, to only partially sort the top_k results
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    return render_template(
           'semantic_search.html',
           value=query,
           result=top_results,
           corpus=corpus,
           cos_scores=cos_scores
    )

# Clustering html
@app.route('/clustering', methods=['POST'])
def Clustering():
    # Center Clusters N
    num_clusters = 5

    # Get user corpus (dynamic)
    value = request.form['corpus']
    value = value.split("\n")
    value = [sen.strip() for sen in value]

    # if number of sentences are over 16, raise error
    over16 = len(value) > 16
    if over16 : return render_template('clustering.html', is_over=over16)

    # Represent Embedding corpus
    corpus_embeddings = embedder.encode(value)

    # Train Clustering Model and fit embedding corpus
    clustering_model = MiniBatchKMeans(n_clusters=num_clusters, random_state=0)
    clustering_model.fit(corpus_embeddings)

    # Grop(clustering) sentences with similar similarities.
    cluster_assignment = clustering_model.labels_
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(value[sentence_id])

    # Get fratures and center in each sentences and K-mean center
    features = clustering_model.transform(corpus_embeddings)
    center = clustering_model.transform(clustering_model.cluster_centers_)

    # Use PCA to reduce dimension
    pca = PCA(n_components=2, random_state=0)
    reduced_features = pca.fit_transform(features)
    reduced_cluster_centers = pca.transform(center)

    # Working to generate image
    everything = np.concatenate((corpus_embeddings, clustering_model.cluster_centers_))

    # Clear previous image log
    plt.cla()

    # Scatter feature's coordinate
    plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        marker='^',
        c=clustering_model.labels_,
        cmap='rainbow',
        label=value)

    # Label the coordinates of the features.
    sen_label = ["Sen"+str(i+1) for i in range(len(reduced_features))]
    for idx, xy in enumerate(reduced_features): plt.text(xy[0]+0.2, xy[1]+0.1, sen_label[idx], fontsize=8)

    # Scatter center's coordinate
    plt.scatter(
        reduced_cluster_centers[:, 0],
        reduced_cluster_centers[:, 1],
        marker='o',
        c='salmon',
        s=80)

    fig = plt.gcf()

    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    return render_template(
           'clustering.html',
           value=value,
           is_over=over16,
           clustered_sentences=clustered_sentences,
           show_image=pngImageB64String)

if __name__ == '__main__':
    # Run app
    app.run(host='0.0.0.0', port='7000', debug=True)
