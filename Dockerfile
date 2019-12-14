FROM frolvlad/alpine-miniconda3

RUN apk add --no-cache nano bash

COPY . /opt/vibe
WORKDIR  /opt/vibe

CMD ["/bin/bash"]
