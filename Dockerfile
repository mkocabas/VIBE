FROM frolvlad/alpine-miniconda3

RUN apk add --no-cache 	nano bash git openssl ca-certificates glib-dev libsm-dev libxrender libxext py3-opengl make \
						autoconf automake libtool gcc g++ libx11-dev libxext-dev mesa-egl glu ffmpeg

COPY . /opt/vibe
WORKDIR /opt/vibe

RUN ./install-libglvnd.sh
RUN pip install -r requirements.txt
RUN ./prepare_data.sh

VOLUME ["/opt/vibe/output", "/opt/vibe/vibe_data"]

CMD ["/bin/bash"]
