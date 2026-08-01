#    This script is part of navis (http://www.github.com/navis-org/navis).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

# The rpy2-based R interface was retired - point people at the file-based
# replacement rather than letting them hit a bare ImportError.
_RETIRED = {
    "r": (
        "The `rpy2`-based R interface (`navis.interfaces.r`) has been retired. "
        "Data exchange with the natverse no longer needs R or `rpy2`: use "
        "`navis.write_rds`/`navis.write_rda` to hand neurons to R "
        "and `navis.read_rds`/`navis.read_rda` to read them back. Its other "
        "functions have native equivalents: `navis.nblast` (was `r.nblast`), "
        "`navis.xform_brain`/`navis.mirror_brain` with navis-flybrains (were "
        "`r.xform_brain`/`r.mirror_brain`) and `navis.read_rda` (was "
        "`r.load_rda`)."
    )
}


def __getattr__(name):
    if name in _RETIRED:
        raise ImportError(_RETIRED[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
