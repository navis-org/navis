import navis
import flybrains

# Add fake bounds for JRCFIB
flybrains.JRCFIB2018Fraw.boundingbox = [0, 34499, 7409, 37539, 2952, 40076]


def test_skeleton_transforms():
    nl = navis.example_neurons(kind='skeleton')

    # Convert example neurons from voxels (raw hemibrain space) to microns
    tr = navis.xform_brain(nl, source='JRCFIB2018Fraw', target='JRCFIB2018Fum')

    assert isinstance(tr, navis.NeuronList)
    assert len(tr) == len(nl)
    assert all(nl.n_nodes == tr.n_nodes)


def test_meshneuron_transforms():
    nl = navis.example_neurons(kind='mesh')

    # Convert example neurons from voxels (raw hemibrain space) to microns
    tr = navis.xform_brain(nl, source='JRCFIB2018Fraw', target='JRCFIB2018Fum')

    assert isinstance(tr, navis.NeuronList)
    assert len(tr) == len(nl)
    assert all(nl.n_vertices == tr.n_vertices)


def test_dotprops_transforms():
    nl = navis.example_neurons(kind='skeleton')
    nl = navis.make_dotprops(nl, k=5)

    # Convert example neurons from voxels (raw hemibrain space) to microns
    tr = navis.xform_brain(nl, source='JRCFIB2018Fraw', target='JRCFIB2018Fum')

    assert isinstance(tr, navis.NeuronList)
    assert len(tr) == len(nl)
    assert all(nl.n_points == tr.n_points)


def test_volume_transforms():
    vol = navis.example_volume('LH')

    # Convert example neurons from voxels (raw hemibrain space) to microns
    tr = navis.xform_brain(vol, source='JRCFIB2018Fraw', target='JRCFIB2018Fum')

    assert isinstance(vol, navis.Volume)
    assert vol.vertices.shape[0] == tr.vertices.shape[0]


def test_skeleton_mirror():
    nl = navis.example_neurons(kind='skeleton')

    # Mirror along midline (produces garbage results but oh well)
    tr = navis.mirror_brain(nl, template='JRCFIB2018Fraw', warp=False)

    assert isinstance(tr, navis.NeuronList)
    assert len(tr) == len(nl)
    assert all(nl.n_nodes == tr.n_nodes)


def test_meshneuron_mirror():
    nl = navis.example_neurons(kind='mesh')

    # Mirror along midline (produces garbage results but oh well)
    tr = navis.mirror_brain(nl, template='JRCFIB2018Fraw', warp=False)

    assert isinstance(tr, navis.NeuronList)
    assert len(tr) == len(nl)
    assert all(nl.n_vertices == tr.n_vertices)


def test_dotprops_mirror():
    nl = navis.example_neurons(kind='skeleton')
    nl = navis.make_dotprops(nl, k=5)

    # Mirror along midline (produces garbage results but oh well)
    tr = navis.mirror_brain(nl, template='JRCFIB2018Fraw', warp=False)

    assert isinstance(tr, navis.NeuronList)
    assert len(tr) == len(nl)
    assert all(nl.n_points == tr.n_points)


def test_volume_mirror():
    vol = navis.example_volume('LH')

    # Mirror along midline (produces garbage results but oh well)
    tr = navis.mirror_brain(vol, template='JRCFIB2018Fraw', warp=False)

    assert isinstance(vol, navis.Volume)
    assert vol.vertices.shape[0] == tr.vertices.shape[0]
