#!/bin/bash

for i in {3..12}; do
  python.exe create_dataset_v2.py ${i}
done
