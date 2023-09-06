import json

import joblib
import numpy as np
from pandas import DataFrame
from sklearn.cluster import HDBSCAN
from umap import UMAP


def reduce_dimensions(vectors: np.array, reducer: UMAP) -> np.array:
    print('Reducing dimensions')
    reduced = reducer.fit_transform(vectors)
    joblib.dump(reducer, '../data/trained_models/umap.joblib')
    return reduced


def cluster_articles(vectors: np.array, clusterer: HDBSCAN) -> np.array:
    print('Clustering data')
    clusters = clusterer.fit_predict(vectors)
    joblib.dump(clusterer, '../data/trained_models/hdbscan.joblib')
    return clusters


def run_cluster_prediction(vector_map: dict[str, list[float]]) -> None:

    article_ids = [id_ for id_ in vector_map.keys()]
    article_vectors = np.array([vec for vec in vector_map.values()])

    umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
    hdbscan = HDBSCAN(min_cluster_size=10)

    reduced_vectors = reduce_dimensions(article_vectors, umap)
    clustered = cluster_articles(reduced_vectors, hdbscan)

    results = DataFrame({'article_id': article_ids, 'cluster_id': clustered})
    filename = f'../data/clustering_results/combined_clusters.csv'
    results.to_csv(filename)


if __name__ == '__main__':
    with open('../data/ada_vectors/en_vectors.json') as f:
        data1 = json.load(f)

    # with open('../data/ada_vectors/es_vectors.json') as f:
    #     data2 = json.load(f)
    #
    # data1.update(data2)
    run_cluster_prediction(data1)
