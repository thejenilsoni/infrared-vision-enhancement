.PHONY: dev api web test lint build docker

dev:
	docker compose up --build

api:
	cd services/inference && uvicorn app.main:app --reload

web:
	npm run dev

test:
	python3 -m unittest discover -s services/inference/tests -v

lint:
	cd services/inference && ruff check app tests
	npm run lint

build:
	npm run build

docker:
	docker compose build
