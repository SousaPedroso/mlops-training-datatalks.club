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
def prepare_features(dataset: Dataset) -> tuple[list, list]:
    tokens = dataset.map(tokenize)["tokens_list"]
    id2word = corpora.Dictionary(tokens)
    corpus = [id2word.doc2bow(token) for token in tokens]

    return id2word, corpus


# 70-20-10 split
@task
def split_dataset(corpus: list, num_index_dictionary: list, output_dir: str):
    # Data will not be shuffled to recover informations from reviews
    train_data, validation_data = train_test_split(corpus, test_size=0.20, shuffle=False)

    train_data, test_data = train_test_split(train_data, test_size=0.125, shuffle=False)

    with open(os.path.join(output_dir, "train.pkl"), "wb") as f_out:
        pickle.dump(train_data, f_out)

    with open(os.path.join(output_dir, "valid.pkl"), "wb") as f_out:
        pickle.dump(validation_data, f_out)

    with open(os.path.join(output_dir, "test.pkl"), "wb") as f_out:
        pickle.dump(test_data, f_out)

    with open(os.path.join(output_dir, "id2word.pkl"), "wb") as f_out:
        pickle.dump(num_index_dictionary, f_out)


@flow(
    name="topic-modeling-training-pipeline-data_preparing",
    task_runner=SequentialTaskRunner(),
)
def prepare_data(input_dir: str, output_dir: str):
    dataset = load_data(input_dir)
    id2w, corpus = prepare_features(dataset)
    split_dataset(corpus, id2w, output_dir)


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
