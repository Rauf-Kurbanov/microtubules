.PHONY: jupyter
jupyter:
	jupyter notebook --no-browser --ip=0.0.0.0 --port=1490 --allow-root --NotebookApp.token=

.PHONY: port
port: PORT=1490
port: LAMBDA_IP=35.188.31.76
port:
	ssh -N -f -L localhost:$(PORT):localhost:$(PORT) rauf@$(LAMBDA_IP)

.PHONY: extract-metadata
extract-metadata:
	python scripts/extract_metadata_new.py \
		--data_root /home/rauf/Data/Ola \
		--meta_path /home/rauf/Data/Ola/metadata/PVE-plateLayouts.csv \
        --flagged_path /home/rauf/Data/Ola/List_Flagged_Images.csv \
        --save_path /home/rauf/Data/Ola/metadata/dataset_filtered.csv