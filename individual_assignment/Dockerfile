FROM python:3.9.13

USER root

RUN apt-get update
RUN apt-get install -y default-jdk

RUN useradd -ms /bin/bash python-api
USER python-api

WORKDIR /repo/
COPY . /repo/

RUN pip install -r /repo/requirements.txt

EXPOSE 8080
ENTRYPOINT ['python']
CMD ["/repo/imdbproject.py"]