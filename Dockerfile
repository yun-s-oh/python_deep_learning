FROM python:3.9.1

ADD . /python-flask
WORKDIR /python-flask
RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=app.py

ENTRYPOINT [ "flask"]
CMD [ "run", "--host", "0.0.0.0", "-p", "5000"]