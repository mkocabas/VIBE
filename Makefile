
## run-demo		:	run demo on sample video.
.PHONY: run-demo
run-demo :
	@python3 demo.py --vid_file sample_video.mp4 --output_folder output/

## docker-build		:	build docker container.
.PHONY: docker-build
docker-build :
	@docker build -t vibe:ubuntu .

## docker-run		:	run docker container.
.PHONY: docker-run
docker-run :
	@docker run -ti -v $(PWD)/output:/opt/vibe/output -v $(PWD)/data:/opt/vibe/vibe_data vibe:ubuntu

## nvidia-build		:	build docker container with nvidia gpu.
.PHONY: nvidia-build
nvidia-build :
	@docker build -t vibe:ubuntu-gpu -f Dockerfile.gpu .

## nvidia-run		:	run docker container with nvidia gpu.
.PHONY: nvidia-run
nvidia-run :
	@nvidia-docker run -ti -v $(PWD)/output:/opt/vibe/output -v $(PWD)/data:/opt/vibe/vibe_data

## help			:	Print commands help.
.PHONY: help
help : Makefile
	@sed -n 's/^##//p' $<

# https://stackoverflow.com/a/6273809/1826109
%:
	@:

