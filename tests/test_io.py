import navis
import pytest
import tempfile

from pathlib import Path


@pytest.mark.parametrize("filename", ['', '{neuron.name}.swc',
                                      'neurons.zip',
                                      '{neuron.name}.swc@neurons.zip'])
def test_swc_io(filename):
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / filename

        # Load example neurons
        n = navis.example_neurons(2, kind='skeleton')

        # Save to file / folder
        navis.write_swc(n, filepath)

        # Load again
        if str(filepath).endswith('.zip'):
            n2 = navis.read_swc(Path(tempdir) / 'neurons.zip')
        else:
            n2 = navis.read_swc(tempdir)

        # Assert that we loaded the same number of neurons
        assert len(n) == len(n2)
