#!/bin/sh
# This script downloads the test data used in tutorials and examples
set -e

# curl wrapper with retries + fail-fast + timeouts so a transient network blip
# on CI doesn't break the build. `-f` makes HTTP errors non-zero (so we retry /
# fail loudly instead of silently writing an error page to the output file).
fetch() {
  curl -fL --retry 5 --retry-all-errors --retry-delay 5 \
       --connect-timeout 30 --max-time 900 "$@"
}

mkdir -p -- "docs/examples/0_io/mmc2"
fetch -o docs/examples/0_io/mmc2/skeletons_swc.zip https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/misc/skeletons_swc.zip

mkdir -p -- "docs/examples/0_io/mmc2/swc/CENT"
fetch -o docs/examples/0_io/mmc2/swc/CENT/11519759.swc https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/misc/11519759.swc

fetch -o docs/examples/0_io/WannerAA201605_SkeletonsGlomeruli.zip https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/misc/WannerAA201605_SkeletonsGlomeruli.zip
cd docs/examples/0_io
unzip -o WannerAA201605_SkeletonsGlomeruli.zip
