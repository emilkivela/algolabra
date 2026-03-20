FROM python:3.12

WORKDIR /algolabra
COPY poetry.lock pyproject.toml ./
RUN pip install poetry==2.2.1
RUN poetry install --no-root


COPY . .


WORKDIR /algolabra/src

ENV FLASK_APP=app
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

EXPOSE 5000

CMD ["poetry", "run", "flask", "run"]
