# MLOps for B2W dataset

**Unfortunately I did not have enough time to complete this project within the cohort period. I will finish it later little by little to improve my knowledge in MLOps.**

Repository for training MLOps for 2023 cohort from [DataTalks.club](https://github.com/DataTalksClub/mlops-zoomcamp). An end-to-end ML pipeline for NLP processing at [B2W dataset](https://huggingface.co/datasets/ruanchaves/b2w-reviews01).

Table of sections:
- [Motivation](#motivation)
- [Structure](#structure)
- [Execution](#execution)

## Motivation

With the use of Machine Learning Operations (MLOps), this project aims to:
- Predict the sentiments of customers
- Model topics to understand customer review content.

Specifically for me, I combined my theoretical knowledge on [NLP](https://github.com/SousaPedroso/NLP) with MLOps to increase my challenge with MLOps, while understading NLP more deeply, mainly for project architecture design.


## Structure

This project is structured as follows:

- [monitoring](/monitoring/) is a folder to organize services responsible for track some aspect of the project (e.g [mlflow](/monitoring/mlflow/) for experiments tracking and [prefect](/monitoring/prefect/) for pipeline orchestration)
- [training](/training/) is a folder to models development for sentimental analysis and topic modelling.

## Requirements

Before execute this project, be sure to attend the requisites:

- docker. You can use this command below for some linux distributions.

```bash
curl -fsSL https://get.docker.com | bash
```

- Python 3.10. You can install through anaconda, venv or directly build from source.


## Execution

To start the services to monitor any modifications, run:

```bash
docker compose --env-file ./monitoring/mlflow/.env  up -d --build
```
