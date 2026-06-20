.PHONY: eda

# Build the data pipeline (bronze -> silver -> gold), then execute the EDA
# notebook end-to-end. Fails if any cell errors. Regenerates all charts.
eda:
	docker compose run --rm jupyter python main.py
	docker compose run --rm jupyter jupyter nbconvert \
		--to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=600 \
		eda/eda.ipynb
	@echo "EDA notebook executed cleanly."