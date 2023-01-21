.PHONY: env
env:
	mamba create -n nagl
	mamba env update -n nagl --file devtools/envs/base.yaml
	conda run --no-capture-output --name nagl pip install --no-deps -e .
	conda run --no-capture-output --name nagl pre-commit install

.PHONY: format
format:
	pre-commit run --all-files