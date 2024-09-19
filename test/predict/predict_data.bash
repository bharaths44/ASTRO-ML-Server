#!/bin/bash

curl -X POST "http://localhost:8080/predict/data" \
    -F "file=@test/subset_30.csv" \
    -F "store_num=1" \
    -F "item_num=1" \
    -F "period_type=M" \
    -F "num_periods=3" \
    -o response.json