FROM python:3.9
WORKDIR /test
COPY . /test
RUN apt-get update && apt-get install -y python3-pip
RUN pip install --trusted-host pypi.python.org -r r.txt
