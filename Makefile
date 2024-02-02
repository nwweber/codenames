.PHONY: hexhex
hexhex:
	ruff format src
	ruff check --fix src
	mypy