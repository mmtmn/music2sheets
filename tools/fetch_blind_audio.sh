#!/usr/bin/env bash
# Download the public-domain recording used for blind test 1 into data/blind/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/blind
curl -L -o data/blind/test1.mp3 \
  "https://archive.org/download/MoonlightSonata_845/Sonata_no_14_in_c_sharp_minor_moonlight_op_27_no_2_Iii.Presto.mp3"
