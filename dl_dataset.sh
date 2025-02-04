#!/bin/bash
rm -rf ./data 
mkdir ./data
curl -L -o ./data/stock_market_dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/jacksoncrow/stock-market-dataset
unzip ./data/stock_market_dataset.zip -d ./data/
rm ./data/stock_market_dataset.zip
python rw_generation.py --display-first 10 --min-sigma 0.3 --min-initial-price 100 --output-dir ./data/GBM_synth/ --num-series 5000
