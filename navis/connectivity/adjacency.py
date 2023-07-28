from typing import Iterable, Optional, NamedTuple
from ..core import TreeNeuron
import networkx as nx
import pandas as pd
import numpy as np
from ..config import get_logger

logger = get_logger(__name__)

OTHER = "__OTHER__"


class Edge(NamedTuple):
    connector_id: int
    source_name: str
    target_name: str
    source_node: Optional[int]
    target_node: Optional[int]


class NeuronConnector:
    """Class which creates a connectivity graph from a set of neurons.

    Connectivity is determined by shared IDs in the ``connectors`` table.

    Add neurons with the `add_neuron` and `add_neurons` methods.
    Alternatively, supply an iterable of neurons in the constructor.
    Neurons must have unique names.

    See the `to_(multi)digraph` method for output.
    """

    def __init__(self, nrns: Optional[Iterable[TreeNeuron]] = None) -> None:
        self.neurons = dict()
        self.connector_xyz = dict()
        # connectors and the treenodes presynaptic to them
        self.conn_inputs = dict()
        # connectors and the treenodes postsynaptic to them
        self.conn_outputs = dict()

        if nrns is not None:
            self.add_neurons(nrns)

    def __len__(self) -> int:
        return len(self.neurons)

    def add_neurons(self, nrns: Iterable[TreeNeuron]):
        """Add several neurons to the connector.

        All neurons must have unique names.

        Parameters
        ----------
        nrns : Iterable[TreeNeuron]

        Returns
        -------
        Modified connector.
        """
        for nrn in nrns:
            self.add_neuron(nrn)
        return self

    def add_neuron(self, nrn: TreeNeuron):
        """Add a single neuron to the connector.

        All neurons must have unique names.

        Parameters
        ----------
        nrn : TreeNeuron

        Returns
        -------
        Modified connector.
        """
        if nrn.name in self.neurons:
            logger.warning(
                "Neuron with name %s has already been added to NeuronConnector. "
                "These will occupy the same node in the graph, "
                "but have connectors from both.",
                nrn.name
            )

        self.neurons[nrn.name] = nrn
        if nrn.connectors is None:
            logger.warning("Neuron with name %s has no connector information", nrn.name)
            return self

        for row in nrn.connectors.itertuples():
            # connector_id, node_id, x, y, z, is_input
            self.connector_xyz[row.connector_id] = (row.x, row.y, row.z)
            if row.type == 1:
                self.conn_outputs.setdefault(row.connector_id, []).append((nrn.name, row.node_id))
            elif row.type == 0:
                if row.connector_id in self.conn_inputs:
                    logger.warning(
                        "Connector with ID %s has multiple inputs: "
                        "connector tables are probably inconsistent",
                        row.connector_id
                    )
                self.conn_inputs[row.connector_id] = (nrn.name, row.node_id)

        return self

    def edges(self, include_other=True) -> Iterable[Edge]:
        """Iterate through all synapse edges.

        Parameters
        ----------
        include_other : bool, optional
            Include edges for which only one partner is known, by default True.
            If included, the name of the unknown partner will be ``"__OTHER__"``,
            and the treenode ID will be None.

        Yields
        ------
        tuple[int, str, str, int, int]
            Connector ID, source name, target name, source treenode, target treenode.
        """
        for conn_id in set(self.conn_inputs).union(self.conn_outputs):
            src, src_node = self.conn_inputs.get(conn_id, (OTHER, None))
            if src_node is None and not include_other:
                continue
            for tgt, tgt_node in self.conn_outputs.get(conn_id, [(OTHER, None)]):
                if tgt_node is None and not include_other:
                    continue
                yield Edge(conn_id, src, tgt, src_node, tgt_node)

    def to_adjacency(self, include_other=True) -> pd.DataFrame:
        """Create an adjacency matrix of neuron connectivity.

        Parameters
        ----------
        include_other : bool, optional
            Whether to include a node called ``"__OTHER__"``,
            which represents all unknown partners.
            By default True.
            This can be helpful when calculating a neuron's input fraction,
            but cannot be used for output fractions if synapses are polyadic.

        Returns
        -------
        pandas.DataFrame
            Row index is source neuron name,
            column index is target neuron name,
            cells are the number of synapses from source to target.
        """
        index = list(self.neurons)
        if include_other:
            index.append(OTHER)
        data = np.zeros((len(index), len(index)), np.uint64)
        df = pd.DataFrame(data, index, index)
        for _, src, tgt, _, _ in self.edges(include_other):
            df[tgt][src] += 1

        return df

    def to_digraph(self, include_other=True) -> nx.DiGraph:
        """Create a graph of neuron connectivity.

        Parameters
        ----------
        include_other : bool, optional
            Whether to include a node called ``"__OTHER__"``,
            which represents all unknown partners.
            By default True.
            This can be helpful when calculating a neuron's input fraction,
            but cannot be used for output fractions if synapses are polyadic.

        Returns
        -------
        nx.DiGraph
            The graph has data ``{"connector_xyz": {connector_id: (x, y, z), ...}}``.
            The nodes have data ``{"neuron": tree_neuron}``.
            The edges have data ``{"connectors": data_frame, "weight": n_connectors}``,
            where the connectors data frame has columns
            "connector_id", "pre_node", "post_node".
        """
        g = nx.DiGraph()
        g.add_nodes_from((k, {"neuron": v}) for k, v in self.neurons.items())
        if include_other:
            g.add_node(OTHER, neuron=None)

        g.graph["connector_xyz"] = self.connector_xyz
        headers = {
            "connector_id": pd.UInt64Dtype(),
            "pre_node": pd.UInt64Dtype(),
            "post_node": pd.UInt64Dtype(),
        }
        edges = dict()
        for conn_id, src, tgt, src_node, tgt_node in self.edges(include_other):
            edges.setdefault((src, tgt), []).append([conn_id, src_node, tgt_node])

        for (src, tgt), rows in edges.items():
            df_tmp = pd.DataFrame(rows, columns=list(headers), dtype=object)
            df = df_tmp.astype(headers, copy=False)
            g.add_edge(src, tgt, connectors=df, weight=len(df))

        return g

    def to_multidigraph(self, include_other=True) -> nx.MultiDiGraph:
        """Create a graph of neuron connectivity where each synapse is an edge.

        Parameters
        ----------
        include_other : bool, optional
            Whether to include a node called ``"__OTHER__"``,
            which represents all unknown partners.
            By default True.
            This can be helpful when calculating a neuron's input fraction,
            but cannot be used for output fractions if synapses are polyadic.

        Returns
        -------
        nx.MultiDiGraph
            The nodes have data ``{"neuron": tree_neuron}``.
            The edges have data
            ``{"pre_node": presyn_treenode_id, "post_node": postsyn_treenode_id, "xyz": connector_location, "connector_id": conn_id}``.
        """
        g = nx.MultiDiGraph()
        g.add_nodes_from((k, {"neuron": v}) for k, v in self.neurons.items())
        if include_other:
            g.add_node(OTHER, neuron=None)

        for conn_id, src, tgt, src_node, tgt_node in self.edges(include_other):
            g.add_edge(
                src,
                tgt,
                pre_node=src_node,
                post_node=tgt_node,
                xyz=self.connector_xyz[conn_id],
                connector_id=conn_id,
            )

        return g
