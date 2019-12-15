
## docker-build		:	build docker container.
.PHONY: docker-build
docker-build :
	@docker build -t vibe:alpine .

## docker-run		:	run docker container.
.PHONY: docker-run
docker-run :
	@docker run -ti -v $(PWD)/output:/opt/vibe/output -v $(PWD)/data:/opt/vibe/vibe_data vibe:alpine

## help			:	Print commands help.
.PHONY: help
help : Makefile
	@sed -n 's/^##//p' $<

# https://stackoverflow.com/a/6273809/1826109
%:
	@:

