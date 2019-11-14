FROM ubuntu
# FROM busybox

EXPOSE 80

RUN apt-get update -qq \
    && apt-get install -y -qq libzmq3-dev

ADD ./server /

CMD ["./server", "80", "accounts.bnk"]