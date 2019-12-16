
## run-demo		:	Run demo on sample video.
.PHONY: run-demo
run-demo :
	@python3 demo.py --vid_file sample_video.mp4 --output_folder output/

## docker-build		:	Build docker container.
.PHONY: docker-build
docker-build :
	@docker build -t mkocabas/vibe:ubuntu .

## docker-run		:	Run docker container.
.PHONY: docker-run
docker-run :
	@docker run -ti -v $(PWD)/output:/opt/vibe/output -v $(PWD)/data:/opt/vibe/vibe_data mkocabas/vibe:ubuntu

## nvidia-build		:	Build docker container with nvidia gpu.
.PHONY: nvidia-build
nvidia-build :
	@docker build -t mkocabas/vibe:ubuntu-gpu -f Dockerfile.gpu .

## nvidia-run		:	Run docker container with nvidia gpu.
.PHONY: nvidia-run
nvidia-run :
	@nvidia-docker run -ti -v $(PWD)/output:/opt/vibe/output -v $(PWD)/data:/opt/vibe/vibe_data mkocabas/vibe:ubuntu-gpu

## help			:	Print commands help.
.PHONY: help
help : Makefile
	@sed -n 's/^##//p' $<

# https://stackoverflow.com/a/6273809/1826109
%:
	@:

