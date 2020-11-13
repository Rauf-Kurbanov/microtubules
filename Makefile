.PHONY: jupyter
jupyter:
	jupyter notebook --no-browser --ip=0.0.0.0 --port=1490 --allow-root --NotebookApp.token=

.PHONY: port
port: PORT=1490
port: LAMBDA_IP=35.188.31.76
port:
	ssh -N -f -L localhost:$(PORT):localhost:$(PORT) rauf@$(LAMBDA_IP)
