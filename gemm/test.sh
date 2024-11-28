#!/bin/bash
rm -rf build/ RGemm.egg-info/
pip install -e .
python test/gemm_test.py