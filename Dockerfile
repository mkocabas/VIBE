FROM ubuntu:19.10

# Install sudo
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
        build-essential \
	bash \
	nano \
	python3 \
        python3-pip \
        python3-setuptools \
	git \
	unzip \
	libglib2.0-0 \
	libsm6 \
	freeglut3-dev \
	ffmpeg \
	wget

COPY . /opt/vibe
WORKDIR /opt/vibe

# RUN ./install-libglvnd.sh
RUN pip3 install -r requirements.txt
RUN ./prepare_data.sh

VOLUME ["/opt/vibe/output", "/opt/vibe/vibe_data"]

CMD ["/bin/bash"]

