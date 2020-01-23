#    This script is part of navis (http://www.github.com/schlegelp/navis).
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

from .. import core


def health_check(x: 'core.NeuronObject', verbose: bool = True) -> None:
    """Run a health check on neuron and flag potential issues.

    Parameters
    ----------
    x :         TreeNeuron | NeuronList
                Neuron(s) whose nodes to classify nodes.
    verbose :   bool
                If True, will print errors in addition to returning them.

    Returns
    -------
    list of issues or None

    """
    if isinstance(x, core.NeuronList):
        for n in x:
            _ = health_check(x)
        return
    elif not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Excepted TreeNeuron/List, got "{type(x)}"')

    issues = []

    # Check if neuron is not a tree
    if not x.is_tree:
        issues.append('is not a tree (networkx.is_forest)')
    # See if there are any cycles
    if x.cycles:
        issues.append(f'has cycles (networkx.find_cycles): {str(x.cycles)}')
    # See if any node has more than one parent
    od = [n[0] for n in x.graph.out_degree if n[1] > 1]
    if od:
        issues.append(f'has nodes with multiple parents (graph.out_degree): {", ".join(od)}')

    if verbose:
        if issues:
            print(f'Neuron {str(x.id)} has issues:')
            for i in issues:
                print(f' - {i}')
        else:
            print(f'Neuron {str(x.id)} seems perfectly fine.')

    return issues if issues else None
