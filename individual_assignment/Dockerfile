FROM python 3.8-slim-buster

RUN useradd -ms /bin/bash ola
USER ola

WORKDIR /repo
COPY . /repo

RUN python -m pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get install -y ant && \
    apt-get clean;

EXPOSE 8080
ENTRYPOINT ['python']
CMD ["python", "/app/main.py"]