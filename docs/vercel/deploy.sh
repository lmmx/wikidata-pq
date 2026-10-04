#!/bin/bash
set -ex

# 1) wget, which Vercel's Amazon Linux images may not include
yum install -y wget

# 2) uv, into ~/.local/bin
wget -qO- https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3) A venv with the Python the build image provides. The docs need only the packages in
#    requirements.txt, not the project (which needs Python 3.13 and the pipeline's
#    dependencies), so the project is not installed.
uv venv
source .venv/bin/activate
python --version

# 4) urllib3<2 first (newer urllib3 breaks on the image's OpenSSL), then the docs packages
uv pip install "urllib3<2"
uv pip install -r docs/vercel/requirements.txt

python -m mkdocs --help && echo $?
