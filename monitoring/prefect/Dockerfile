FROM prefecthq/prefect:2-python3.10

RUN pip install -U pip && pip install pipenv

WORKDIR /opt/prefect

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy
