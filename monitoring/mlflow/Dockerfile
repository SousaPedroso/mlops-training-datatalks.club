FROM python:3.10.12-slim

RUN pip install -U pip && pip install pipenv

WORKDIR /mlflow/

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy
