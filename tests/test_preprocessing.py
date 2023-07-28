import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
# pylint: disable=wrong-import-position
from datasets import Dataset

# pylint: disable=import-error
from training.preprocessing import tokenize


def test_tokenize_empty_text():
    ds_dict = {"review_text": [""]}
    ds = Dataset.from_dict(ds_dict)
    expected_output_dict = {"review_text": [""], "tokens_list": [" "]}
    expected_output = Dataset.from_dict(expected_output_dict)
    cur_result = ds.map(tokenize)
    assert cur_result.data.equals(expected_output.data)

    ds_dict = {"review_text": [" \n a b c d e f g h i j k l m n o p q r s t u v w x y z"]}
    ds = Dataset.from_dict(ds_dict)
    # pylint: disable=line-too-long
    expected_output_dict = {
        "review_text": [" \n a b c d e f g h i j k l m n o p q r s t u v w x y z"],
        "tokens_list": [" "],
    }
    expected_output = Dataset.from_dict(expected_output_dict)
    cur_result = ds.map(tokenize)
    assert cur_result.data.equals(expected_output.data)


def test_tokenize_stopwords_and_punctuation():
    # single corpus
    ds_dict = {"review_text": ["Este é um teste de pré-processamento de texto."]}
    ds = Dataset.from_dict(ds_dict)
    expected_output_dict = {
        "review_text": ["Este é um teste de pré-processamento de texto."],
        "tokens_list": ["test pré process text"],
    }
    expected_output = Dataset.from_dict(expected_output_dict)
    cur_result = ds.map(tokenize)
    assert cur_result.data.equals(expected_output.data)

    # multiple corpus
    ds_dict = {
        "review_text": [
            "Este é um teste de pré-processamento de texto.",
            "Este é, + um teste de pré-processamento de texto.!!! Ok?",
        ]
    }
    ds = Dataset.from_dict(ds_dict)
    expected_output_dict = {
        "review_text": [
            "Este é um teste de pré-processamento de texto.",
            "Este é, + um teste de pré-processamento de texto.!!! Ok?",
        ],
        "tokens_list": [
            "test pré process text",
            "test pré process text ok",
        ],
    }
    expected_output = Dataset.from_dict(expected_output_dict)
    cur_result = ds.map(tokenize)

    assert cur_result.data.equals(expected_output.data)


def test_tokenize_numbers():
    # single corpus
    ds_dict = {"review_text": ["Este é um teste com números: 123."]}
    ds = Dataset.from_dict(ds_dict)
    expected_output_dict = {
        "review_text": ["Este é um teste com números: 123."],
        "tokens_list": ["test númer"],
    }
    expected_output = Dataset.from_dict(expected_output_dict)
    cur_result = ds.map(tokenize)
    assert cur_result.data.equals(expected_output.data)

    # multiple corpus
    ds_dict = {
        "review_text": [
            "Vcs esta1 de parabens, o at3ndimento foi otimo",
            "O at3ndim3nto para nós 4 foi de primeira, pode ter ctz 6vão crescer mais",
        ]
    }
    ds = Dataset.from_dict(ds_dict)
    expected_output_dict = {
        "review_text": [
            "Vcs esta1 de parabens, o at3ndimento foi otimo",
            "O at3ndim3nto para nós 4 foi de primeira, pode ter ctz 6vão crescer mais",
        ],
        "tokens_list": [
            "vcs esta1 parabens at3ndiment otim",
            "at3ndim3nt primeir pod ter ctz 6vã cresc",
        ],
    }
    expected_output = Dataset.from_dict(expected_output_dict)
    cur_result = ds.map(tokenize)
    assert cur_result.data.equals(expected_output.data)
