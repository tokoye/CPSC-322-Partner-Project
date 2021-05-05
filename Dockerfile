From continuumio/anaconda3:2020.11

ADD . /code
WORKDIR /code

ENTRYPOINT ["phython", "SpotifyPartner_app"]