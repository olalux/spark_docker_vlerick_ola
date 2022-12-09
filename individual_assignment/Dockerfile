FROM python 3.8-slim-buster

RUN useradd -ms /bin/bash ola
USER ola

WORKDIR /repo
COPY . /repo

RUN python -m pip install -r requirements.txt

EXPOSE 8080
ENTRYPOINT ['python']
CMD ['main.py']