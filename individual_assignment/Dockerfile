FROM python:3.9.13

RUN useradd -ms /bin/bash ola
USER ola

WORKDIR /repo
COPY . /repo

RUN python -m pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y default-jdk

EXPOSE 8080
ENTRYPOINT ['python']
CMD ["python", "/app/main.py"]