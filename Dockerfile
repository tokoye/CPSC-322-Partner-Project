FROM continuumio/anaconda3:2020.11

ADD . /code
WORKDIR /code
ENTRYPOINT ["python", "interview_flask_app.py"]