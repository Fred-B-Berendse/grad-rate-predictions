FROM ubuntu:latest
LABEL maintainer="fred.b.berendse@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
EXPOSE 8105
CMD ["app.py"]