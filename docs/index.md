---
icon: material/home
hide:
  - navigation
  - toc
---


<div id="neuron-viewer" style="width: 100%; height: 400px; border-radius: 16px; overflow: hidden; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); background-image: url('_static/logo_new_banner.jpg'); background-size: cover; background-position: center; background-repeat: no-repeat;"></div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.159.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.159.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

function initViewer() {
  const container = document.getElementById('neuron-viewer');
  if (!container) return;

  const width = container.clientWidth;
  const height = container.clientHeight;

  // Scene setup
  const scene = new THREE.Scene();
  scene.background = null;

  const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
  camera.position.z = 300;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.domElement.style.borderRadius = '16px';
  renderer.domElement.style.display = 'block';
  container.appendChild(renderer.domElement);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
  directionalLight.position.set(10, 20, 10);
  scene.add(directionalLight);

  let neuronModel = null;
  let neuronPivot = null;

  function frameModelToView(model) {
    const box = new THREE.Box3().setFromObject(model);
    const sphere = box.getBoundingSphere(new THREE.Sphere());

    const fov = THREE.MathUtils.degToRad(camera.fov);
    const hFov = 2 * Math.atan(Math.tan(fov / 2) * camera.aspect);
    const distanceV = sphere.radius / Math.sin(fov / 2);
    const distanceH = sphere.radius / Math.sin(hFov / 2);
    const distance = Math.max(distanceV, distanceH) * 0.75;

    camera.position.set(0, 0, distance);
    camera.near = Math.max(distance / 1000, 0.01);
    camera.far = distance * 100;
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }

  // Load model
  const loader = new GLTFLoader();
  const modelPath = '_static/neuron.glb';

  loader.load(modelPath, function(gltf) {
    const model = gltf.scene;

    // Apply materials to all meshes
    const neuronMaterial = new THREE.MeshPhongMaterial({
      color: 0x000000,
      side: THREE.DoubleSide,
      flatShading: false
    });

    model.traverse((child) => {
      if (child.isMesh) {
        child.material = neuronMaterial;
      }
    });

    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    model.position.sub(center);

    neuronPivot = new THREE.Group();
    neuronPivot.position.set(0, 0, 0);
    neuronPivot.add(model);

    neuronModel = model;
    scene.add(neuronPivot);
    frameModelToView(neuronPivot);
  }, undefined, function(error) {
    console.error('Error loading neuron model:', error);
    console.error('Attempted to load from:', modelPath);
  });

  // Mouse controls
  let mouseDown = false;
  let mouseX = 0, mouseY = 0;
  let targetRotationX = 0, targetRotationY = 0;
  let currentRotationX = 0, currentRotationY = 0;
  const autoRotateSpeed = 0.0025;

  container.addEventListener('mousedown', (e) => {
    mouseDown = true;
    mouseX = e.clientX;
    mouseY = e.clientY;
  });

  container.addEventListener('mousemove', (e) => {
    if (!mouseDown) return;
    const deltaX = e.clientX - mouseX;
    const deltaY = e.clientY - mouseY;
    targetRotationY += deltaX * 0.01;
    targetRotationX += deltaY * 0.01;
    mouseX = e.clientX;
    mouseY = e.clientY;
  });

  container.addEventListener('mouseup', () => {
    mouseDown = false;
  });

  container.addEventListener('mouseleave', () => {
    mouseDown = false;
  });

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);

    if (!mouseDown) {
      targetRotationY += autoRotateSpeed;
    }

    currentRotationX += (targetRotationX - currentRotationX) * 0.1;
    currentRotationY += (targetRotationY - currentRotationY) * 0.1;

    if (neuronPivot) {
      neuronPivot.rotation.x = currentRotationX;
      neuronPivot.rotation.y = currentRotationY;
    }

    renderer.render(scene, camera);
  }
  animate();

  // Handle window resize
  window.addEventListener('resize', () => {
    const newWidth = container.clientWidth;
    const newHeight = container.clientHeight;
    camera.aspect = newWidth / newHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(newWidth, newHeight);

    if (neuronPivot) {
      frameModelToView(neuronPivot);
    }
  });
}

// Initialize viewer when ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initViewer);
} else {
  initViewer();
}
</script>

# <span style="color:rgb(255,190,40);font-weight:bold">N</span>euron <span style="color:rgb(255,190,40);font-weight:bold">A</span>nalysis and <span style="color:rgb(255,190,40);font-weight:bold">Vis</span>ualization

[![Docs](https://github.com/navis-org/navis/actions/workflows/build-docs.yml/badge.svg)](https://github.com/navis-org/navis/actions/workflows/build-docs.yml) [![Tests](https://github.com/navis-org/navis/actions/workflows/test-package.yml/badge.svg)](https://github.com/navis-org/navis/actions/workflows/test-package.yml) [![Test tutorials](https://github.com/navis-org/navis/actions/workflows/test-tutorials.yml/badge.svg)](https://github.com/navis-org/navis/actions/workflows/test-tutorials.yml) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navis-org/navis/blob/master/examples/colab.ipynb) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8191725.svg)](https://zenodo.org/doi/10.5281/zenodo.4699382) [![Downloads](https://pepy.tech/badge/navis)](https://pepy.tech/project/navis)

{{ navis }} is a Python library for analysis and visualization of neuron
morphology. It stands on the shoulders of the excellent
[`natverse`](http://natverse.org) for R.



---

**[Features](#features)** - **[Quickstart](quickstart.md)** - **[Installation](installation.md)**

<div id="navis-carousel" style="position: relative; width: 100%; max-width: 900px; margin: 1rem auto 1.25rem auto; border-radius: 8px; overflow: hidden;">
  <div id="carousel-images-container"></div>
  <div id="carousel-caption" style="position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.6); color: white; padding: 12px 16px; font-size: 14px; text-align: center; opacity: 1; transition: opacity 0.8s ease-in-out;"></div>
  <button id="carousel-prev" style="position: absolute; left: 10px; top: 50%; transform: translateY(-50%); background: rgba(0,0,0,0.4); color: white; border: none; font-size: 24px; padding: 8px 12px; border-radius: 4px; cursor: pointer; z-index: 10; transition: background 0.2s;" onmouseover="this.style.background='rgba(0,0,0,0.7)'" onmouseout="this.style.background='rgba(0,0,0,0.4)'">❮</button>
  <button id="carousel-next" style="position: absolute; right: 10px; top: 50%; transform: translateY(-50%); background: rgba(0,0,0,0.4); color: white; border: none; font-size: 24px; padding: 8px 12px; border-radius: 4px; cursor: pointer; z-index: 10; transition: background 0.2s;" onmouseover="this.style.background='rgba(0,0,0,0.7)'" onmouseout="this.style.background='rgba(0,0,0,0.4)'">❯</button>
</div>

<script>
(async function () {
  try {
    const response = await fetch('_static/image_carousel/manifest.json');
    const data = await response.json();
    const imageContainer = document.getElementById('carousel-images-container');
    const captionEl = document.getElementById('carousel-caption');
    const prevBtn = document.getElementById('carousel-prev');
    const nextBtn = document.getElementById('carousel-next');

    imageContainer.style.position = 'relative';

    if (!data.images || !data.images.length) return;

    // Shuffle images array (Fisher-Yates algorithm)
    for (let i = data.images.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [data.images[i], data.images[j]] = [data.images[j], data.images[i]];
    }

    // Create image elements dynamically
    data.images.forEach((item, index) => {
      const file = typeof item === 'string' ? item : item.file;
      const caption = typeof item === 'string' ? '' : (item.caption || '');

      const imgEl = document.createElement('img');
      imgEl.src = '_static/image_carousel/' + file;
      imgEl.alt = caption || file.replace(/\.[^.]+$/, '');
      imgEl.className = 'navis-carousel-slide';
      imgEl.dataset.caption = caption;
      imgEl.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 280px; object-fit: contain; opacity: ${index === 0 ? '1' : '0'}; transition: opacity 0.8s ease-in-out;`;
      imageContainer.appendChild(imgEl);
    });

    // Create spacer div
    const spacer = document.createElement('div');
    spacer.style.cssText = 'width: 100%; height: 280px;';
    imageContainer.appendChild(spacer);

    const slides = document.querySelectorAll('#navis-carousel .navis-carousel-slide');
    let current = 0;
    let autoPlayInterval;

    function showSlide(index) {
      slides.forEach((slide, i) => {
        slide.style.opacity = i === index ? '1' : '0';
      });
      captionEl.textContent = slides[index].dataset.caption || '';
    }

    function resetAutoPlay() {
      clearInterval(autoPlayInterval);
      autoPlayInterval = setInterval(() => {
        current = (current + 1) % slides.length;
        showSlide(current);
      }, 2800);
    }

    prevBtn.addEventListener('click', () => {
      current = (current - 1 + slides.length) % slides.length;
      showSlide(current);
      resetAutoPlay();
    });

    nextBtn.addEventListener('click', () => {
      current = (current + 1) % slides.length;
      showSlide(current);
      resetAutoPlay();
    });

    // Show initial caption
    if (slides.length > 0) {
      captionEl.textContent = slides[0].dataset.caption || '';
    }

    resetAutoPlay();
  } catch (error) {
    console.error('Error loading carousel manifest:', error);
  }
})();
</script>


## Features

<div class="grid cards" markdown>

-   :simple-databricks:{ .lg .middle } __Polgyglot__

    ---

    Support for all kinds of [neuron types](generated/gallery/tutorial_basic_01_neurons): skeletons, meshes, dotprops and images.

-   :material-eye:{ .lg .middle } __Exploration__

    ---

    Designed to let you explore your data interactively from Jupyter notebooks,
    terminal or via scripts.

-   :fontawesome-solid-circle-notch:{ .lg .middle } __Analysis__

    ---

    Calculate Strahler indices, cable length, volume, tortuosity, NBLAST
    and many other [morphometrics](generated/gallery/2_morpho/tutorial_morpho_01_analyze).

-   :fontawesome-solid-brush:{ .lg .middle } __Visualization__

    ---

    Generate beautiful, publication-ready 2D (matplotlib) and 3D (octarine,
    vispy or plotly) [figures](generated/gallery/#plotting).

-   :material-progress-wrench:{ .lg .middle } __Processing__

    ---

    Smoothing, resampling, skeletonization, meshing and [more](api.md#neuron-morphology)!

-   :fontawesome-solid-computer:{ .lg .middle } __Fast__

    ---

    Uses compiled Rust code under-the-hood and
    out-of-the-box support for [multiprocessing](generated/gallery/6_misc/tutorial_misc_00_multiprocess).

-   :material-lightbulb-group:{ .lg .middle } __Clustering__

    ---

    Cluster your neurons by e.g. morphology using [NBLAST](generated/gallery/5_nblast/tutorial_nblast_00_intro).

-   :material-move-resize:{ .lg .middle } __Transforms__

    ---

    Fully featured [transform system](generated/gallery/6_misc/tutorial_misc_01_transforms) to move neurons between brain spaces.
    We support CMTK, Elastix, landmark-based transforms and more!

-   :octicons-file-directory-symlink-24:{ .lg .middle } __Import/Export__

    ---

    Read and write from/to SWC, NRRD, Neuroglancer's precomputed format,
    OBJ, STL and [more](generated/gallery/#import-export)!

-   :octicons-globe-24:{ .lg .middle } __Connected__

    ---

    Load neurons straight from Allen's
    [MICrONS](generated/gallery/4_remote/tutorial_remote_02_microns) datasets,
    [neuromorpho](http://neuromorpho.org), [neuPrint](generated/gallery/4_remote/tutorial_remote_00_neuprint),
    the [H01 dataset](generated/gallery/4_remote/tutorial_remote_04_h01)
    or any NeuroGlancer source.

-   :material-connection:{ .lg .middle } __Interfaces__

    ---

    Load neurons into [Blender 3D](generated/gallery/3_interfaces/tutorial_interfaces_02_blender), simulate neurons and networks using
    [NEURON](generated/gallery/3_interfaces/tutorial_interfaces_00_neuron), or use the R natverse library via `rpy2`.

-   :material-google-circles-extended:{ .lg .middle } __Extensible__

    ---

    Write your own library built on top of NAVis functions. See
    our [ecosystem](ecosystem.md) for examples.

</div>

Check out the [Tutorials](generated/gallery/) and [API reference](api.md) to see
what you can do with {{ navis }}.

Need help? Use [discussions](https://github.com/navis-org/navis/discussions)
on Github to ask questions!

{{ navis }} is licensed under the GNU GPL v3+ license. The source code is hosted
at [Github](https://github.com/navis-org/navis). Feedback, feature requests
and bug reports are very welcome and best placed in a
[Github issue](https://github.com/navis-org/navis/issues)