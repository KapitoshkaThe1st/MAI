FROM ubuntu

EXPOSE 80

RUN apt-get update -qq \
    && apt-get install -y -qq libzmq3-dev

ADD ./server /

RUN ls -l

CMD ["bash", "./server", "80"]