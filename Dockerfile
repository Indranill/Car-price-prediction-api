FROM python:3.7.13-slim

WORKDIR /flask-docker

COPY requirenment.txt requirenment.txt

#RUN apt-get -y install python3-pip

RUN python3 -m pip install --upgrade pip

RUN pip install -r requirenment.txt

COPY ./ ./

EXPOSE 80

CMD ["python3", "app.py"]
