## Machine Learning

These tutorials show you how to prepare neurons as inputs for machine-learning
models. Neurons are variable-sized graphs and meshes living in arbitrary poses
and physical units - most models want fixed-size, canonically-posed tensors. The
`navis.ml` helpers bridge that gap:

1. **Normalize** a neuron's pose so the model doesn't have to learn that a
   shifted/rotated/rescaled neuron means the same thing.
2. Turn neurons into **fixed-size model inputs** - point clouds with features, or
   evenly-sized fragments for batching.
3. **Augment** a training set with realistic perturbations to make models robust
   to nuisance variation.
