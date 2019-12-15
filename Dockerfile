FROM frolvlad/alpine-miniconda3

RUN apk add --no-cache nano bash git openssl ca-certificates glib-dev libsm-dev libxrender libxext py3-opengl make libtool gcc g++ libx11-dev libxext-dev mesa-egl glu ffmpeg

COPY . /opt/vibe
WORKDIR  /opt/vibe

RUN pip install -r requirements.txt

CMD ["/bin/bash"]
