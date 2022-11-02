# [START dockerfile]
FROM python:3.7-slim
RUN pip install flask
RUN pip install pandas
WORKDIR /mlops_test
COPY . /mlops_test
RUN apt-get update && apt-get install -y python3-pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt
# [END dockerfile]
