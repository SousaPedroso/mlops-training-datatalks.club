"""
Orchestration of topical modeling of customer content reviews
"""
from argparse import ArgumentParser

from prefect import flow
from prefect.task_runners import SequentialTaskRunner
from preprocess import prepare_data
from train import train


@flow(name="topic-modeling-training-pipeline", task_runner=SequentialTaskRunner())
def main(
    input_dir: str,
    output_dir: str,
    experiment_name: str,
    passes: int,
    topics: list,
    alpha: list,
    beta: list,
):
    prepare_data(input_dir, output_dir)

    train(experiment_name, output_dir, passes, topics, alpha, beta)


# pylint: disable=duplicate-code
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Data's repository to be used for processing",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Data's path to save the preprocessed data",
        required=True,
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment to be used for tracking",
        required=True,
    )

    parser.add_argument(
        "--passes",
        type=int,
        help="Number of passes to be used for model training",
        default=3,
    )

    parser.add_argument(
        "--topics",
        type=list,
        help=(
            "Number of topics to be used for model training.            Expected three values:"
            " start, stop, step"
        ),
        default=[5, 20, 5],
    )

    parser.add_argument(
        "--alpha",
        type=list,
        help="Alpha hyperparameter to be used for model training",
        default=[0.01, 0.31, 0.61, 0.91],
    )

    parser.add_argument(
        "--beta",
        type=list,
        help="Beta hyperparameter to be used for model training",
        default=[0.01, 0.31, 0.61, 0.91],
    )

    args = parser.parse_args()

    main(
        args.input_dir,
        args.output_dir,
        args.experiment_name,
        args.passes,
        args.topics,
        args.alpha,
        args.beta,
    )
