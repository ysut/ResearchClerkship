#!/bin/bash

set -eu

python main.py \
	--input /Downloads/merged.gene2.header.tsv \
	--resources /bulk \
	--gq 20 \
	--dp 10 \
	--ad 0.3 \
	--ggm 0.02 \
	--cadd 15 \
	--revel 0.3 \
	--spliceai 0.1

