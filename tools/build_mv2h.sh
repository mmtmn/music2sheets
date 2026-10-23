#!/usr/bin/env bash
# Fetch and compile MV2H (McLeod & Steedman, ISMIR 2018) into third_party/MV2H. Needs a JDK.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
if [ ! -d third_party/MV2H ]; then
  git clone --depth 1 https://github.com/apmcleod/MV2H.git third_party/MV2H
fi
make -C third_party/MV2H
echo "MV2H built in third_party/MV2H/bin"
