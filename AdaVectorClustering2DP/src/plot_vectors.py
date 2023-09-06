import json

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.graph_objects as go


def reduce_dimensions(vector_map: dict[str, list[float]]) -> pd.DataFrame:
    print('Creating 2D projections')

    article_ids = [id_ for id_ in vector_map.keys()]
    article_vectors = np.array([vec for vec in vector_map.values()])

    tsne = TSNE(n_components=2, random_state=0, perplexity=40)
    projections = tsne.fit_transform(article_vectors)

    projections_frame = pd.DataFrame(projections)
    projections_frame.columns = ['dimension_1', 'dimension_2']
    projections_frame.index = article_ids

    return projections_frame


def plot_clusters(two_d_projections: np.array, hover_text: np.array,
                  cluster_ids: np.array) -> None:
    print('Plotting')
    plot = go.Figure(
        data=[
            go.Scatter(
                x=two_d_projections[:, 0],
                y=two_d_projections[:, 1],
                mode='markers',
                marker_color=cluster_ids,
                marker_colorscale='spectral',
                text=hover_text
            )
        ]
    )
    plot.update_layout(
        autosize=False,
        width=1000,
        height=800,
        # hoverlabel=dict(font_color="black", bgcolor="white"),
    )
    plot.show()


def run_plotting(article_data_path: str, vector_path: str, cluster_path: str) -> None:
    with open(vector_path) as f:
        vector_map = json.load(f)

    two_d_projections = reduce_dimensions(vector_map)

    article_data = pd.read_csv(article_data_path)
    clusters = pd.read_csv(cluster_path)

    # Merge data to ensure order
    combined_data = \
        article_data.merge(clusters, left_on='article_id', right_on='article_id')
    combined_data = \
        combined_data.merge(two_d_projections, left_on='article_id', right_index=True)
    combined_data = combined_data.set_index('article_id', drop=True)

    # Remove "noise"
    combined_data = combined_data[combined_data['cluster_id'] != -1]

    projections = combined_data[['dimension_1', 'dimension_2']].to_numpy()
    plot_clusters(projections, combined_data['title'], combined_data['cluster_id'])


if __name__ == '__main__':
    root = '../data'
    run_plotting(
        f'{root}/raw_articles/en_articles_1692777557_1692863957.csv',
        f'{root}/ada_vectors/en_vectors.json',
        f'{root}/clustering_results/en_clusters.csv'
    )
