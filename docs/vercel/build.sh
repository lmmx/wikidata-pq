#!/bin/bash
set -ex

# The venv made by deploy.sh
source .venv/bin/activate
python --version

mkdocs build
