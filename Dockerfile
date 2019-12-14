FROM frolvlad/alpine-miniconda3

RUN apk add --no-cache nano bash git openssl ca-certificates

COPY . /opt/vibe
WORKDIR  /opt/vibe

RUN pip install -r requirements.txt

CMD ["/bin/bash"]
