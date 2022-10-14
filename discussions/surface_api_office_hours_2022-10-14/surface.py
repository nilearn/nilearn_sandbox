from __future__ import annotations
import dataclasses
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import nilearn
import nilearn.surface
from nilearn import datasets
from nilearn import plotting


SurfArray = Union[np.ndarray, pathlib.Path, str]
PolyData = Mapping[str, SurfArray]


class Mesh:
    n_nodes: int = dataclasses.field(init=False)
    _raw_geometry: Any

    def __init__(self, geometry: Any) -> None:
        self._raw_geometry = geometry
        self.n_nodes = nilearn.surface.load_surf_mesh(
            self._raw_geometry
        ).coordinates.shape[0]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {self.n_nodes} nodes>"

    # `coordinates` and `faces` are defined as properties now for compatibility
    # with current nilearn plotting API; but as loading the data can take time
    # or fail it would probably be better if they were regular functions.

    @property
    def coordinates(self) -> np.ndarray:
        return nilearn.surface.load_surf_mesh(self._raw_geometry).coordinates

    @property
    def faces(self) -> np.ndarray:
        return nilearn.surface.load_surf_mesh(self._raw_geometry).faces


PolyMesh = Mapping[str, Mesh]
PolyMeshFamily = Mapping[str, PolyMesh]


def _get_mesh_dims(mesh: PolyMesh) -> Dict[str, int]:
    return {
        part_name: mesh_part.n_nodes for (part_name, mesh_part) in mesh.items()
    }


@dataclasses.dataclass()
class SurfaceImage:
    data: PolyData
    meshes: PolyMeshFamily
    shape: Tuple[int, ...] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        data_shapes = {k: _load(v).shape for (k, v) in self.data.items()}
        self._check_shapes(data_shapes)
        n_nodes = sum(s[-1] for s in data_shapes.values())
        ds = list(data_shapes.values())[0]
        if len(ds) == 1:
            self.shape = (n_nodes,)
        else:
            assert len(ds) == 2
            self.shape = (ds[0], n_nodes)

    def _check_shapes(
        self, data_shapes: Mapping[str, Tuple[int, ...]]
    ) -> None:
        mesh_dims = _get_mesh_dims(list(self.meshes.values())[0])
        for mesh in self.meshes.values():
            assert _get_mesh_dims(mesh) == mesh_dims
        assert {k: v[-1] for (k, v) in data_shapes.items()} == mesh_dims

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.shape}>"


def _load(data: SurfArray) -> np.ndarray:
    return nilearn.surface.load_surf_data(data)


class SurfaceMasker:
    mask_img_: Optional[SurfaceImage]

    def _fit_mask_img(self, img: SurfaceImage) -> None:
        mask_data = {
            k: np.ones(_load(v).shape[-1], dtype=bool)
            for (k, v) in img.data.items()
        }
        self.mask_img_ = SurfaceImage(data=mask_data, meshes=img.meshes)  # type: ignore

    def fit(self, img: SurfaceImage, y: Any = None) -> SurfaceMasker:
        del y
        self._fit_mask_img(img)
        assert self.mask_img_ is not None
        start, stop = 0, 0
        self.slices = {}
        for part_name, mask in self.mask_img_.data.items():
            assert isinstance(mask, np.ndarray)
            stop = start + mask.sum()
            self.slices[part_name] = start, stop
            start = stop
        self.output_dim = stop
        return self

    def transform(self, img: SurfaceImage) -> np.ndarray:
        assert self.mask_img_ is not None
        output = np.empty(
            (_load(list(img.data.values())[0]).shape[0], self.output_dim)
        )
        for part_name, (start, stop) in self.slices.items():
            mask = self.mask_img_.data[part_name]
            assert isinstance(mask, np.ndarray)
            output[:, start:stop] = _load(img.data[part_name])[:, mask]
        return output

    def fit_transform(self, img: SurfaceImage, y: Any = None) -> np.ndarray:
        del y
        return self.fit(img).transform(img)

    def inverse_transform(self, masked_images: np.ndarray) -> SurfaceImage:
        assert self.mask_img_ is not None
        is_2d = len(masked_images.shape) > 1
        masked_images = np.atleast_2d(masked_images)
        data = {}
        for part_name, mask in self.mask_img_.data.items():
            assert isinstance(mask, np.ndarray)
            data[part_name] = np.zeros(
                (masked_images.shape[0], mask.shape[0]),
                dtype=masked_images.dtype,
            )
            start, stop = self.slices[part_name]
            data[part_name][:, mask] = masked_images[:, start:stop]
            if not is_2d:
                data[part_name] = data[part_name].squeeze()
        return SurfaceImage(data=data, meshes=self.mask_img_.meshes)  # type: ignore


class SurfaceLabelsMasker:
    labels_img: SurfaceImage
    labels_data_: np.ndarray
    labels_: np.ndarray
    label_names_: np.ndarray

    def __init__(
        self,
        labels_img: SurfaceImage,
        label_names: Optional[Mapping[Any, str]] = None,
    ) -> None:
        self.labels_img = labels_img
        self.labels_data_ = np.concatenate(
            [_load(data_part) for data_part in labels_img.data.values()]
        )
        all_labels = set(self.labels_data_.ravel())
        all_labels.discard(0)
        self.labels_ = np.asarray(list(all_labels))
        if label_names is None:
            self.label_names_ = np.asarray(
                [str(label) for label in self.labels_]
            )
        else:
            self.label_names_ = np.asarray(
                [label_names[label] for label in self.labels_]
            )

    def fit(
        self, img: Optional[SurfaceImage] = None, y: Any = None
    ) -> SurfaceLabelsMasker:
        del img, y
        return self

    def transform(self, img: SurfaceImage) -> np.ndarray:
        img_data = np.hstack(
            [
                np.atleast_2d(_load(data_part))
                for data_part in img.data.values()
            ]
        )
        output = np.empty((img_data.shape[0], len(self.labels_)))
        for i, label in enumerate(self.labels_):
            output[:, i] = img_data[:, self.labels_data_ == label].mean(axis=1)
        return output

    def fit_transform(self, img: SurfaceImage, y: Any = None) -> np.ndarray:
        del y
        return self.fit(img).transform(img)

    def inverse_transform(self, masked_images: np.ndarray) -> SurfaceImage:
        is_2d = len(masked_images.shape) > 1
        masked_images = np.atleast_2d(masked_images)
        data = {}
        for part_name, labels_part in self.labels_img.data.items():
            labels_part = _load(labels_part)
            data[part_name] = np.zeros(
                (masked_images.shape[0], labels_part.shape[0]),
                dtype=masked_images.dtype,
            )
            for label_idx, label in enumerate(self.labels_):
                data[part_name][:, labels_part == label] = masked_images[
                    :, label_idx
                ]
            if not is_2d:
                data[part_name] = data[part_name].squeeze()
        return SurfaceImage(data=data, meshes=self.labels_img.meshes)  # type: ignore


def load_fsaverage(mesh_name: str = "fsaverage5") -> PolyMeshFamily:
    fsaverage = datasets.fetch_surf_fsaverage(mesh_name)
    meshes: Dict[str, Dict[str, Mesh]] = {}
    renaming = {"pial": "pial", "white": "white_matter", "infl": "inflated"}
    for mesh_type, mesh_name in renaming.items():
        meshes[mesh_name] = {}
        for hemisphere in "left", "right":
            meshes[mesh_name][f"{hemisphere}_hemisphere"] = Mesh(
                fsaverage[f"{mesh_type}_{hemisphere}"]
            )
    return meshes


def fetch_nki(n_subjects=1) -> Sequence[SurfaceImage]:
    fsaverage = load_fsaverage("fsaverage5")
    nki_dataset = datasets.fetch_surf_nki_enhanced(n_subjects=n_subjects)
    images = []
    for left, right in zip(
        nki_dataset["func_left"], nki_dataset["func_right"]
    ):
        left_data = nilearn.surface.load_surf_data(left).T
        right_data = nilearn.surface.load_surf_data(right).T
        img = SurfaceImage(
            {"left_hemisphere": left_data, "right_hemisphere": right_data},
            meshes=fsaverage,
        )
        images.append(img)
    return images


def fetch_destrieux() -> Tuple[SurfaceImage, Dict[int, str]]:
    fsaverage = load_fsaverage("fsaverage5")
    destrieux = datasets.fetch_atlas_surf_destrieux()
    label_names = {
        i: label.decode("utf-8") for (i, label) in enumerate(destrieux.labels)
    }
    return (
        SurfaceImage(
            {
                "left_hemisphere": destrieux["map_left"],
                "right_hemisphere": destrieux["map_right"],
            },
            meshes=fsaverage,
        ),
        label_names,
    )


def plot_surf_img(
    img: SurfaceImage,
    parts: Optional[Sequence[str]] = None,
    mesh: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    if mesh is None:
        mesh = list(img.meshes.keys())[0]
    if parts is None:
        parts = list(img.data.keys())
    fig, axes = plt.subplots(
        1,
        len(parts),
        subplot_kw={"projection": "3d"},
        figsize=(4 * len(parts), 4),
    )
    for ax, mesh_part in zip(axes, parts):
        plotting.plot_surf(
            img.meshes[mesh][mesh_part],
            img.data[mesh_part],
            axes=ax,
            title=mesh_part,
            **kwargs,
        )
    assert isinstance(fig, plt.Figure)
    return fig
