"""
Script to prepare the data for topic modeling
"""
import os
import pickle
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

# pylint: disable=[wrong-import-position, import-error]
from argparse import ArgumentParser

from datasets import Dataset, load_dataset
from gensim import corpora
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from preprocessing import tokenize
from sklearn.model_selection import train_test_split


@task(retries=5)
def load_data(data_repo: str) -> Dataset:
    return load_dataset(data_repo, split="train")


@task
def prepare_features(dataset: Dataset) -> tuple[list, list, list]:
    aux_list = ["foo" for _ in range(len(dataset))]
    dataset = dataset.add_column(name="tokens_list", column=aux_list)
    dataset = dataset.map(tokenize)
    tokens = [tokens.split() for tokens in dataset["tokens_list"]]
    id2word = corpora.Dictionary(tokens)
    corpus = [id2word.doc2bow(token) for token in tokens]

    return id2word, corpus, tokens


# 70-20-10 split
@task
def split_dataset(corpus: list, tokens: list, num_index_dictionary: list, output_dir: str):
    # Data will not be shuffled to recover informations from reviews
    train_data_corpus, validation_data_corpus, *splitted_tokens = train_test_split(
        corpus, tokens, test_size=0.20, shuffle=False
    )

    train_data_tokens, validation_data_tokens = splitted_tokens[0], splitted_tokens[1]
    train_data_corpus, test_data_corpus, *splitted_tokens = train_test_split(
        train_data_corpus, train_data_tokens, test_size=0.125, shuffle=False
    )
    train_data_tokens, test_data_tokens = splitted_tokens[0], splitted_tokens[1]

    with open(os.path.join(output_dir, "train_corpus.pkl"), "wb") as f_out:
        pickle.dump(train_data_corpus, f_out)

    with open(os.path.join(output_dir, "valid_corpus.pkl"), "wb") as f_out:
        pickle.dump(validation_data_corpus, f_out)

    with open(os.path.join(output_dir, "train_tokens.pkl"), "wb") as f_out:
        pickle.dump(train_data_tokens, f_out)

    with open(os.path.join(output_dir, "valid_tokens.pkl"), "wb") as f_out:
        pickle.dump(validation_data_tokens, f_out)

    with open(os.path.join(output_dir, "test_corpus.pkl"), "wb") as f_out:
        pickle.dump(test_data_corpus, f_out)

    with open(os.path.join(output_dir, "test_tokens.pkl"), "wb") as f_out:
        pickle.dump(test_data_tokens, f_out)

    with open(os.path.join(output_dir, "id2word.pkl"), "wb") as f_out:
        pickle.dump(num_index_dictionary, f_out)


@flow(
    name="topic-modeling-training-pipeline-data_preparing",
    task_runner=SequentialTaskRunner(),
)
def prepare_data(input_dir: str, output_dir: str):
    dataset = load_data(input_dir)
    id2w, corpus, tokens = prepare_features(dataset)
    split_dataset(corpus, tokens, id2w, output_dir)


# pylint: disable=duplicate-code
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Data's path to be used for data preparation",
        required=True,
    )

    parser.add_argument(
        "--output_dir", type=str, help="Data's path to save the preprocessed data", required=True
    )

    args = parser.parse_args()

    prepare_data(args.input_dir, args.output_dir)
