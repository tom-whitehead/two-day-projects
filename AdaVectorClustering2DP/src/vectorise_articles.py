import json
import math
import os
import time

from langchain.embeddings import OpenAIEmbeddings
from pandas import read_csv, DataFrame

MAX_CHARACTERS = 1500
BATCH_SIZE = 1000


def init_model():
    return OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_TOKEN'))


def trim_article(text: str) -> str:
    if len(text) < MAX_CHARACTERS:
        return text

    text = text[:MAX_CHARACTERS]
    index_final_stop = text.rfind('.')
    if index_final_stop == -1:
        return text

    return text[:index_final_stop + 1]


def vectorise_articles(model: OpenAIEmbeddings, article_texts: list[str]) -> list[list[float]]:
    article_texts = [trim_article(text) for text in article_texts]
    return model.embed_documents(article_texts)


def generate_batches(article_data: DataFrame) -> DataFrame:
    n_batches = math.ceil(len(article_data) / BATCH_SIZE)
    for batch in range(n_batches):
        print(f"Processing batch {batch + 1} of {n_batches}")
        start_index = batch * BATCH_SIZE
        end_index = start_index + BATCH_SIZE
        yield article_data.iloc[start_index:end_index]


def run_vectorisation(article_path: str) -> None:
    article_data = read_csv(article_path)
    model = init_model()
    vector_map = {}

    # Run in batches to avoid rate limiting
    for article_batch in generate_batches(article_data):
        vectors = vectorise_articles(model, article_batch['text'].tolist())
        vector_map_batch = \
            {id_: vector for id_, vector in zip(article_batch['article_id'].tolist(), vectors)}
        vector_map.update(vector_map_batch)

        sleep_seconds = 20
        print(f"Sleeping for {sleep_seconds} seconds")
        time.sleep(sleep_seconds)

    output_path = f'{"/".join(article_path.split("/")[:2])}/ada_vectors/' \
                  f'{article_path.split("/")[-1][:2]}_vectors.json'
    with open(output_path, 'w') as f:
        json.dump(vector_map, f)


if __name__ == '__main__':
    run_vectorisation('../data/raw_articles/en_articles_1692777557_1692863957.csv')
