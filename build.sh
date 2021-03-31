#!/bin/bash

python3 setup.py build_ext --inplace
rm -r build/
rm topic_models/sample/_sample.cpp
rm topic_models/utils/bigdouble.cpp