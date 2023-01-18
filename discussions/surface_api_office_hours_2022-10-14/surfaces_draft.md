---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Notes from 2022-10-14 

## Done

Drop `MultiMeshFamily`: each `SurfaceImage` has exactly 1 mesh (typically the pial surface for one of the fsaverage resolutions).
When we want to plot a map on a different mesh, construct a new image.

Therefore `SurfaceImage` looks like:

~~~python

class SurfaceImage:
    data: PolyData
    meshes: PolyMesh
    shape: Tuple[int, ...]
~~~

<!-- #md -->
```
<SurfaceImage (895, 20484)>:
  data:
    left_hemisphere: <ndarray (895, 10242)>
    right_hemisphere: <ndarray (895, 10242)>
  mesh:
    left_hemisphere: <Mesh with 10242 nodes>
    right_hemisphere: <Mesh with 10242 nodes>
  shape: (895, 20484)
```
<!-- #endmd -->

When calling `vol_to_surf` with 2 surfaces (for sampling between pial and wm surface), the outer (pial) surface is kept unless the user explicitly says otherwise.




# Surface images in Nilearn

Current state:

- low-level functions for reading and plotting data for 1 mesh at a time that work well
- sampling a volume at the nodes of 1 mesh
- no real support for analysis of surface images because no representation of the surface data for a whole brain

Relevant issues

- links in [this issue](https://github.com/nilearn/nilearn/issues/2681)
- [draft surface labels masker PR](https://github.com/nilearn/nilearn/issues/2423) labels masker dealing with 1 mesh /surface map at a time

## Main analysis steps

- Reading images
- Masking
- Analysis
- Un-masking
- Plotting
- Writing images

## Examples

### Masking and un-masking

```python
from nilearn import plotting

import surface

img = surface.fetch_nki()[0]
print(f"NKI image: {img}")

masker = surface.SurfaceMasker()
masked_data = masker.fit_transform(img)
print(f"Masked data shape: {masked_data.shape}")

mean_data = masked_data.mean(axis=0)
mean_img = masker.inverse_transform(mean_data)
print(f"Image mean: {mean_img}")

surface.plot_surf_img(mean_img)
plotting.show()
```

### Connectivity with a surface atlas and `SurfaceLabelsMasker`

```python
from nilearn import plotting
from nilearn import connectome

import surface


img = surface.fetch_nki()[0]
print(f"NKI image: {img}")

labels_img, label_names = surface.fetch_destrieux()
print(f"Destrieux image: {labels_img}")
surface.plot_surf_img(labels_img, cmap="gist_ncar", avg_method="median")

labels_masker = surface.SurfaceLabelsMasker(labels_img, label_names).fit()
masked_data = labels_masker.transform(img)
print(f"Masked data shape: {masked_data.shape}")

connectome = (
    connectome.ConnectivityMeasure(kind="correlation").fit([masked_data]).mean_
)
plotting.plot_matrix(connectome, labels=labels_masker.label_names_)

plotting.show()
```


### Using the `Decoder`

```python
from nilearn import plotting, decoding
from nilearn._utils import param_validation
import nilearn.decoding.decoder
import numpy as np

import surface
```

The following is just disabling a couple of checks performed by the decoder that would force us to use a `NiftiMasker`.

```python
def monkeypatch_masker_checks():
    def adjust_screening_percentile(screening_percentile, *args, **kwargs):
        return screening_percentile

    param_validation._adjust_screening_percentile = adjust_screening_percentile

    def check_embedded_nifti_masker(estimator, *args, **kwargs):
        return estimator.mask

    decoding.decoder._check_embedded_nifti_masker = check_embedded_nifti_masker


monkeypatch_masker_checks()
```

Now using the appropriate masker we can use a `Decoder` on surface data just as we do for volume images.

```python
img = surface.fetch_nki()[0]
y = np.random.RandomState(0).choice([0, 1], replace=True, size=img.shape[0])

decoder = decoding.Decoder(
    mask=surface.SurfaceMasker(),
    param_grid={"C": [0.01, 0.1]},
    cv=3,
    screening_percentile=1,
)
decoder.fit(img, y)
print("CV scores:", decoder.cv_scores_)

surface.plot_surf_img(decoder.coef_img_[0], threshold=1e-6)
plotting.show()
```

### Decoding with a scikit-learn `Pipeline`

```python
from nilearn import plotting
import numpy as np
from sklearn import pipeline, feature_selection, linear_model, preprocessing

import surface


img = surface.fetch_nki()[0]
y = np.random.RandomState(0).normal(size=img.shape[0])

decoder = pipeline.make_pipeline(
    surface.SurfaceMasker(),
    preprocessing.StandardScaler(),
    feature_selection.SelectKBest(
        score_func=feature_selection.f_regression, k=500
    ),
    linear_model.Ridge(),
)
decoder.fit(img, y)

coef_img = decoder[:-1].inverse_transform(np.atleast_2d(decoder[-1].coef_))


vmax = max([np.absolute(dp).max() for dp in coef_img.data.values()])
surface.plot_surf_img(
    coef_img,
    cmap="cold_hot",
    vmin=-vmax,
    vmax=vmax,
    threshold=1e-6,
)
plotting.show()
```

## API

### Surface images

#### Summary

This is the representation for the NKI resting state image used in the examples (time-series with 895 points, each hemisphere has 10242 nodes, the meshes are not stored in memory):

<!-- #md -->
```
<SurfaceImage (895, 20484)>:
  data:
    left_hemisphere: <ndarray (895, 10242)>
    right_hemisphere: <ndarray (895, 10242)>
  mesh:
    left_hemisphere: <Mesh with 10242 nodes>
    right_hemisphere: <Mesh with 10242 nodes>
  shape: (895, 20484)
```
<!-- #endmd -->

#### Details

A few simple type definitions for the basic components.
A `SurfArray` is a Numpy array or a path to a `.gii` or Freesurfer file.
(Could also be called `SurfData`, `Data` -- "array" makes it clear it contains a single array)

~~~python
SurfArray = Union[np.ndarray, pathlib.Path, str]
~~~

`PolyData` is a `dict` mapping (brain part, hemisphere) names to data arrays.
Other names could be `MultiData` (but "Multi" is used for other stuff in nilearn), `CompositeData`, `PolyArray`.

~~~python
PolyData = Mapping[str, SurfArray]
example = {"left_hemisphere": "left.gii", "right_hemisphere": "right.gii"}
~~~

> If we are sure that there will only ever be 2 surfaces in a brain corresponding to 2 hemispheres we could have something less flexible, like:
>
> ~~~python
> class SurfaceData:
>     left_hemisphere: SurfArray
>     right_hemisphere: SurfArray
> ~~~

`Mesh` is basically the same as `nilearn.surface.Mesh`; we could use that instead.
However, we do not want to force having the coordinates & faces loaded in memory (but we do probably want the number of nodes).
Also, creating `Mesh` with `namedtuple` is not an ideal choice IMO because:
- the iterable interface doesn't seem very appropriate because there is no obvious order -- should coordinates or faces go first, same applies to `Surface`.
- from discussions `Mesh` was introduced it seems there is some confusion around `namedtuple`.

~~~python
class Mesh:
    n_nodes: int = dataclasses.field(init=False)

    def coordinates(self) -> np.ndarray:
        """load the array of node coordinates (n_nodes, 3)"""

    def faces(self) -> np.ndarray:
        """load the array of faces (n_faces, 3)"""
~~~


A `PolyMesh` is similar to `PolyData` for meshes: it maps (hemisphere) names to meshes.

~~~python
PolyMesh = Mapping[str, Mesh]
example = {"left_hemisphere": Mesh("left.gii"), "right_hemisphere": Mesh("right.gii")}
~~~


> As for `PolyData`, this could be more rigid if we only ever want 2 hemispheres.
> In the other direction, we can choose more general names and types if we want to allow mixing surface and volume data (eg 3D data for the cerebellum).
> In that case we could have something like
> ~~~python
> PolyGeometry = Mapping[str, Union[Mesh, AffineAndShape]]
> # or
> PolyGeometry = Mapping[str, Union[Mesh, NiftiHeader]]
> ~~~

A `PolyMeshFamily` (could be `PolyMeshCollection`, ...) is a `PolyMesh` declined for several surface types such as pial, white matter, inflated, etc.
~~~python
PolyMeshFamily = Mapping[str, PolyMesh]
example = {
    "pial": {"left_hemisphere": "...", "right_hemisphere": "..."},
    "wihte matter": {"left_hemisphere": "...", "right_hemisphere": "..."},
}
~~~


These dictionaries can be replaced with user-defined types to add functionality and enforce invariants such as having the same keys (brain parts) in all the members of a `PolyMeshFamily`.

Finally a `SurfaceImage` contains a `PolyData` (a (1 or 2)-dimensional array for each hemisphere) and a `PolyMesh` (with at least one element).
The keys in `data` and `mesh` must match; they will probably always be `left_hemisphere` and `right_hemisphere`.
The shape is `(n_time_points, total_number_of_mesh_nodes)` for 2d images, or `(total_number_of_mesh_nodes,)` for 1-D images.
We could (should?) also transpose that (ie first dimension is always number of nodes), following the signal-processing convention rather than the stats/ML convention -- as is done in Nifti.
It depends if we prefer to be similar to Nifti images or to Maskers' output.

~~~python

class SurfaceImage:
    data: PolyData
    mesh: PolyMesh
    shape: Tuple[int, ...] = dataclasses.field(init=False)
~~~

#### Keeping meshes out of memory

Meshes are not needed for most analysis steps beyond checking that shapes are compatible, so we just store the file path and load them when needed (eg for plotting or in init to store their shape).

The main use case (I think?) is using one of the fsaverage meshes, so we could have a way of representing these without a concrete file path, such as

~~~python
mesh = "nilearn:fsaverage5"
~~~

or

```python
import enum


class StandardMesh(enum.Enum):
    FSAVERAGE_5 = enum.auto()
    FSAVERAGE_7 = enum.auto()
    # ...
```


Benefits:

- nilearn can know metadata (eg shapes) for these meshes without reading them
- easier to be sure several copies of the same mesh are not loaded in memory
- no need to know where a mesh is (eg in `$NILEARN_DATA`), or it can even be missing and fetched if necessary
- no need to have a 1:1 mapping between meshes and files
- possibly beter `__repr__` & autocompletion

### Maskers

Maskers do not perform resampling to different meshes, but they check that (mask, labels, maps) images and (functional) images to transform have compatible shapes.
They could do the same filtering detrending etc. as volume maskers, and even simple spatial smoothing.
They operate on `SurfaceImage` and produce Numpy arrays (containing data from both hemispheres) so to the rest of the pipeline a surface image and a surface masker together provide the same interface as a Nifti image and a Nifti masker as shown in examples above.
We would also need the same utilities as for volume images such as:

- concatenating images
- indexing image time series
- a surface equivalent of `new_img_like`


We should also have a `vol_to_surf` function that produces a `SurfaceImage`.

### Plotting

High-level plotting functions accept a full `SurfaceImage` (and possibly a second one for the background).
Low-level plotting functions working with a single data array and mesh can be kept or made private after a deprecation cycle.

### Reading & writing

Hopefully in the future this will be handled by nibabel; we need to check that the representations we consider are not too far from their plan.

Reading:

- Unless there is a standard users will probably need to provide an explicit mapping of hemisphere name to data file for each image
- There could be helpers for outputs of widely used tools like freesurfer
- We can think of ways to provide patterns for loading multiple images eg for multiple subjects

Writing:

- we probably don't want to deal with file formats, until nibabel does it we can consider creating a directory to save an image
- look for some standard or a widely used tool whose output we can imitate
- to avoid storing meshes several times we could use symlinks
