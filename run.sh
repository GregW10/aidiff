#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python main256_nolog.py

if [ $? -eq 0 ]; then
	mv -fv losses.csv "$(ls -td model_*/ | head -1)"
	exit 0
fi

mv -fv losses.csv "$(ls -td model_*/ | head -1)"

while true; do
	output="$(ls -t */*.pth)"
	if [ $? -eq 0 ]; then
		latest_model=$(echo $output | awk '{ print $1 }')
		echo "LOADING LATEST MODEL:" "$latest_model"
		python main256_nolog.py "$latest_model"
	else
		python main256_nolog.py
	fi
	if [ $? -eq 0 ]; then
		mv -fv losses.csv "$(ls -td model_*/ | head -1)"
		exit 0
	fi
	mv -fv losses.csv "$(ls -td model_*/ | head -1)"
	sleep 1
done
