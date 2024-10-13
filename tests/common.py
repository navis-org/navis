import warnings

try:
    import igraph
except ModuleNotFoundError:
    igraph = None
    warnings.warn('iGraph library not found. Will test only with NetworkX.')

import navis


def with_igraph(func):
    def wrapper(*args, **kwargs):
        navis.config.use_igraph = False
        res1 = func(*args, **kwargs)
        if igraph:
            navis.config.use_igraph = True
            res2 = func(*args, **kwargs)
            assert res1 == res2
        return res1
    return wrapper
