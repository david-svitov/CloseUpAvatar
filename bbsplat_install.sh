#!/bin/bash

pip install ./3rd_party/diff-bbsplat-rasterization
pip uninstall -y pytorch3d
pip install git+https://github.com/facebookresearch/pytorch3d.git
