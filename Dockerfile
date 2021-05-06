FROM continuumio/anaconda3:2020.11

ADD . /code
WORKDIR /code
ENTRYPOINT ["python", "track_data_app.py"]