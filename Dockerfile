# [START dockerfile]
FROM python:3.7-slim
RUN pip install flask
WORKDIR /mlops_test
COPY hello_app.py /mlops_test/hello_app.py
ENTRYPOINT ["python"]
CMD ["/mlops_test/hello_app.py"]
# [END dockerfile]
