import os
import sys
import navis

import plotly.io as pio
from plotly.io._sg_scraper import plotly_sg_scraper

pio.renderers.default = 'sphinx_gallery'
# pio.renderers.default = "sphinx_gallery_png"

# This makes sure we don't have any ugly progress bars in the examples
navis.config.pbar_hide = True

from mkdocs_gallery.gen_gallery import DefaultResetArgv

min_reported_time = 0
if 'SOURCE_DATE_EPOCH' in os.environ:
    min_reported_time = sys.maxint if sys.version_info[0] == 2 else sys.maxsize

# To be used as the "base" config,
# this script is referenced in the mkdocs.yaml under `conf_script` option: docs/gallery_conf.py
conf = {
    'reset_argv': DefaultResetArgv(),
    'min_reported_time': min_reported_time,
    # "image_scrapers": ('matplotlib', plotly_sg_scraper),
}

