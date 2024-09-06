"""
This file defines the macros for the mkdocs site.
"""

import navis

# Some variables

# Depth from the
DEPTH = None

def define_env(env):
    @env.macro
    def autosummary(func):
        """Return first line of dosctring"""
        try:
            f = navis
            for name in func.split('.'):
                if name == 'navis':
                    continue
                f = getattr(f, name)
            return f.__doc__.split('\n')[0]
        except BaseException as e:
            return f'Error finding docstring for {func}: {e}'