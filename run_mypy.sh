#!/bin/sh
export MYPYPATH=./stubs
mypy --allow-redefinition --python-executable python3 navis/
