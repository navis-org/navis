#!/bin/sh
# This script downloads the test data used in tutorials and examples
mkdir -p -- "docs/examples/0_io/mmc2"
curl -o docs/examples/0_io/mmc2/skeletons_swc.zip https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/misc/skeletons_swc.zip

mkdir -p -- "docs/examples/0_io/mmc2/swc/CENT"
curl -o docs/examples/0_io/mmc2/swc/CENT/11519759.swc https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/misc/11519759.swc

curl -o docs/examples/0_io/WannerAA201605_SkeletonsGlomeruli.zip https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/misc/WannerAA201605_SkeletonsGlomeruli.zip
cd docs/examples/0_io
unzip WannerAA201605_SkeletonsGlomeruli.zip