FROM frolvlad/alpine-miniconda3

RUN apk add --no-cache nano bash

CMD ["/bin/bash"]
