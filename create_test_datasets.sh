#!/bin/bash

for i in {3..12}; do
  python.exe create_test_dataset.py $i
done
