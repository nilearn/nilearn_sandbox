"""
Script to generate fsaverage gifti files for fsaverage resolutions 3 to 7.

Freesurfer needs to be installed to run this script,
and the env variable FREESURFER_HOME set to a valid value.
"""

# %%
import argparse
import os
import subprocess
from pathlib import Path

import nibabel
import numpy as np
from nilearn import datasets, surface
from scipy.sparse import coo_matrix, lil_matrix
from tqdm import tqdm

# %%
# UTIL FUNCTIONS


def mesh_to_graph(coordinates, triangles):
    """Compute adjacency matrix of a given mesh."""
    n_points = coordinates.shape[0]
    edges = np.hstack(
        (
            np.vstack((triangles[:, 0], triangles[:, 1])),
            np.vstack((triangles[:, 0], triangles[:, 2])),
            np.vstack((triangles[:, 1], triangles[:, 0])),
            np.vstack((triangles[:, 1], triangles[:, 2])),
            np.vstack((triangles[:, 2], triangles[:, 0])),
            np.vstack((triangles[:, 2], triangles[:, 1])),
        )
    )
    weights = np.ones(edges.shape[1])

    # Divide data by 2 since all edges i -> j are counted twice
    # because they all belong to exactly two triangles on the mesh
    connectivity = (
        coo_matrix((weights, edges), (n_points, n_points)).tocsr() / 2
    )

    # Making it symmetrical
    connectivity = (connectivity + connectivity.T) / 2
    connectivity.data[connectivity.data > 0] = 1

    return connectivity


# Compute adjacency matrices of order d a_d and af_d of fs7 and fs7 flat resp.
def get_shortest_path_matrix(connectivity, d_max=10):
    """Compute powers of the connectivity matrix.

    If A is the adjacency matrix, A^n_{i,j} denotes
    the number of paths of size n from vertex i to vertex j.
    Here we use B = A.astype("bool") such that B^n_{i,j} is True
    if and only if there exists a path of size n from vertex i to vertex j.
    """
    connectivity_boolean = connectivity.astype("bool")

    power_acc = connectivity_boolean.copy()
    shortest_path_matrix = lil_matrix(power_acc).astype(np.int32)

    for d in tqdm(range(2, d_max), desc="Shortest path power", leave=False):
        power_acc = np.dot(power_acc, connectivity_boolean)
        # Update the shortest_path matrix such that
        # SP_{t+1,i,j} = SP_{t,i,j} if SP_{t,i,j} > 0 else d * PA_{t+1,i,j}

        current_distance = d * power_acc

        shortest_path_matrix = (
            shortest_path_matrix.multiply(shortest_path_matrix > 0)
            + current_distance
            - current_distance.multiply(shortest_path_matrix > 0)
        )

        # Don't forget to set the diagonal to 0 after that.
        shortest_path_matrix.setdiag(0)

    return shortest_path_matrix


# Faces to keep
def should_keep_face(face, diff):
    """Determine if a mesh face should be filtered out."""
    if diff[face[0], face[1]] > 0:
        return False
    elif diff[face[0], face[2]] > 0:
        return False
    elif diff[face[1], face[2]] > 0:
        return False
    return True


# %%
if __name__ == "__main__":
    # %%
    # Parse arguments
    parser = argparse.ArgumentParser(
        description=(
            "Generate tarballs for freesurfer surfaces files used in nilearn"
        ),
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to output folder where generated tarballs will be written",
    )
    parser.add_argument(
        "flat_surfaces_dir",
        type=str,
        help=(
            "Path to folder containing flat surfaces for fsaverage7 named"
            " flat_{left,right}.gii"
        ),
    )

    args = parser.parse_args()
    output_path = Path(args.output_dir)
    flat_surfaces_path = Path(args.flat_surfaces_dir)

    # %%
    # output_path = Path(
    #     "/home/alexis/singbrain/outputs/_036_generate_fsaverage_gifti_files_all_resolutions/new3"
    # )
    # flat_surfaces_path = Path(
    #     "/home/alexis/singbrain/data/fsaverage_flat/surfaces"
    # )

    # %%
    # Global variables
    fs_surfaces_path = Path(os.environ.get("FREESURFER_HOME")) / "subjects"
    nilearn_data_path = Path(datasets.utils.get_data_dirs()[0])

    # %%
    resolutions = [
        "fsaverage",  # fsaverage7
        "fsaverage6",
        "fsaverage5",
        "fsaverage4",
        "fsaverage3",
    ]

    attribute_to_filename = {
        "area_left": "lh.area",
        "area_right": "rh.area",
        "curv_left": "lh.curv",
        "curv_right": "rh.curv",
        "infl_left": "lh.inflated",
        "infl_right": "rh.inflated",
        "pial_left": "lh.pial",
        "pial_right": "rh.pial",
        "sphere_left": "lh.sphere",
        "sphere_right": "rh.sphere",
        "sulc_left": "lh.sulc",
        "sulc_right": "rh.sulc",
        "thick_left": "lh.thickness",
        "thick_right": "rh.thickness",
        "white_left": "lh.white",
        "white_right": "rh.white",
    }

    # %%
    for resolution in tqdm(resolutions, desc="Resolution", leave=False):
        resolution_folder = output_path / resolution
        resolution_folder.mkdir(parents=True, exist_ok=True)

        left_reference_surface = (
            fs_surfaces_path / resolution / "surf" / "lh.pial"
        )
        right_reference_surface = (
            fs_surfaces_path / resolution / "surf" / "rh.pial"
        )

        for attribute, filename in tqdm(
            attribute_to_filename.items(), desc="Mesh", leave=False
        ):
            input_file = fs_surfaces_path / resolution / "surf" / filename
            output_gii_file = output_path / resolution / f"{attribute}.gii"

            # Convert file from freesurfer format to gifti
            if any(
                word in str(input_file)
                for word in ["infl", "pial", "sphere", "white"]
            ):
                # Convert surface file
                convert_process = subprocess.Popen(
                    ["mris_convert", input_file, output_gii_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            else:
                # Convert scalar overlay
                convert_process = subprocess.Popen(
                    [
                        "mris_convert",
                        "-c",
                        input_file,
                        (
                            left_reference_surface
                            if "lh" in filename
                            else right_reference_surface
                        ),
                        output_gii_file,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            convert_process.wait()

            # Compress gifti file
            with open(f"{output_gii_file}.gz", "w") as f:
                gzip_process = subprocess.Popen(
                    ["gzip", "-c", output_gii_file],
                    stdout=f,
                    stderr=subprocess.PIPE,
                )
            gzip_process.wait()

            # Remove unzipped file
            os.remove(output_gii_file)

    # %%
    # Generate flat maps
    high_res_fsaverage = None
    high_res_flat_mesh = None
    high_res_connectivity = None
    high_res_flat_connectivity = None

    # %%
    for hemi in ["left", "right"]:
        for resolution in resolutions:
            # %%
            curr_res_fsaverage = surface.load_surf_mesh(
                datasets.fetch_surf_fsaverage(mesh=resolution)[f"pial_{hemi}"]
            )
            curr_res_connectivity = mesh_to_graph(
                curr_res_fsaverage[0], curr_res_fsaverage[1]
            )

            if resolution in ["fsaverage4", "fsaverage3"]:
                # For resolutions >= 5, the order of vertices
                # is consistent across resolutions
                # (for instance the n vertices of fsaverage5
                # are in the same order as the first
                # n vertices of fsaverage6).
                # For lower resolutions, vertices need to be reordered.
                curr_res_fsaverage_matches_in_high_res_fsaverage = [
                    np.argwhere(
                        [
                            np.allclose(
                                vertex,
                                high_res_fsaverage[0][i, :],
                                atol=2 * 1e-2,
                            )
                            for vertex in curr_res_fsaverage[0]
                        ]
                    )
                    for i in tqdm(
                        range(curr_res_fsaverage[0].shape[0]),
                        desc="Vertices in higher resolution",
                        leave=False,
                    )
                ]

                # Check that only one matching vertex was found per vertex
                assert np.sum(
                    np.array(
                        list(
                            map(
                                lambda x: x.shape[0],
                                curr_res_fsaverage_matches_in_high_res_fsaverage,
                            )
                        )
                    )
                ) == len(curr_res_fsaverage_matches_in_high_res_fsaverage)

                new_order = np.array(
                    curr_res_fsaverage_matches_in_high_res_fsaverage
                ).flatten()

                new_order_inverted = np.empty_like(new_order)
                new_order_inverted[new_order] = np.arange(new_order.size)

                # Reorder current fsaverage
                faces_reordered = np.vectorize(
                    lambda x: new_order_inverted[x]
                )(curr_res_fsaverage[1]).astype(np.int32)
                curr_res_fsaverage = (
                    curr_res_fsaverage[0][new_order],
                    faces_reordered,
                )

            # %%
            if resolution == "fsaverage":
                fs_hemi = "lh" if hemi == "left" else "rh"
                curr_res_flat_mesh = surface.load_surf_data(
                    str(flat_surfaces_path / f"flat_{fs_hemi}.gii")
                )
            else:
                # Generate connectivity matrices
                # of fsaverage and flat fsaverage surfaces
                # for higher resolution
                high_res_shortest_path_matrix = get_shortest_path_matrix(
                    high_res_connectivity, d_max=10
                )
                high_res_flat_shortest_path_matrix = get_shortest_path_matrix(
                    high_res_flat_connectivity, d_max=10
                )

                # Keep current resolution faces
                # based on wether the shortest path
                # remained constant for all edges of the face.
                # Indeed, if it is higher in the flat map,
                # it means this face was removed in the flat map.
                diff_shortest_paths = (
                    high_res_shortest_path_matrix
                    - high_res_flat_shortest_path_matrix
                )

                selected_faces = np.array(
                    [
                        face
                        for face in tqdm(
                            curr_res_fsaverage[1],
                            desc="Select face",
                            leave=False,
                        )
                        if should_keep_face(face, diff_shortest_paths)
                    ]
                )

                # Generate flat mesh for current resolution.
                # Coordinates are deduced from that of the
                # higher resolution mesh.
                # Faces are those that have just been selected.
                if resolution in ["fsaverage4", "fsaverage3"]:
                    selected_faces_reordered = np.vectorize(
                        lambda x: new_order[x]
                    )(selected_faces).astype(np.int32)
                    curr_res_flat_mesh = (
                        high_res_flat_mesh[0][
                            : curr_res_fsaverage[0].shape[0]
                        ][new_order_inverted],
                        selected_faces_reordered,
                    )

                    # Revert order of vertices in current fsaverage mesh
                    faces_rereordered = np.vectorize(lambda x: new_order[x])(
                        curr_res_fsaverage[1]
                    ).astype(np.int32)
                    curr_res_fsaverage = (
                        curr_res_fsaverage[0][new_order_inverted],
                        faces_rereordered,
                    )
                else:
                    curr_res_flat_mesh = (
                        high_res_flat_mesh[0][
                            : curr_res_fsaverage[0].shape[0]
                        ],
                        selected_faces,
                    )

            # Save flat mesh as gifti image
            flat_mesh_img = nibabel.gifti.gifti.GiftiImage()
            flat_mesh_img.add_gifti_data_array(
                nibabel.gifti.gifti.GiftiDataArray(
                    curr_res_flat_mesh[0], "NIFTI_INTENT_POINTSET"
                )
            )
            flat_mesh_img.add_gifti_data_array(
                nibabel.gifti.gifti.GiftiDataArray(
                    curr_res_flat_mesh[1], "NIFTI_INTENT_TRIANGLE"
                )
            )

            nibabel.save(
                flat_mesh_img,
                str(output_path / resolution / f"flat_{hemi}.gii.gz"),
            )

            high_res_fsaverage = curr_res_fsaverage
            high_res_flat_mesh = curr_res_flat_mesh
            high_res_connectivity = mesh_to_graph(
                curr_res_fsaverage[0], curr_res_fsaverage[1]
            )
            high_res_flat_connectivity = mesh_to_graph(
                curr_res_flat_mesh[0], curr_res_flat_mesh[1]
            )

    # %%
    # Create tarball from all generated files
    for resolution in resolutions:
        tarball_process = subprocess.Popen(
            f"tar -czvf {output_path / f'{resolution}.tar.gz'} *.gii.gz",
            cwd=output_path / resolution,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    # %%
    # CHECKS
    # Mostly check that generated files are equal files
    # that are already shipped with nilearn

    # %%
    # Make sure meshes have been downloaded for all fsaverage resolutions
    for resolution in tqdm(resolutions):
        meshes = datasets.fetch_surf_fsaverage(mesh=resolution)

    # %%
    resolution = resolutions[0]
    mesh_attribute = list(attribute_to_filename)[0]

    # %%
    for resolution in tqdm(resolutions, desc="Resolution"):
        for mesh_attribute in tqdm(
            attribute_to_filename, desc="Mesh", leave=False
        ):
            # %%
            # mesh_type, hemi = mesh_attribute.split("_")
            old_mesh_path = (
                nilearn_data_path / resolution / f"{mesh_attribute}.gii.gz"
            )
            new_mesh_path = (
                output_path / resolution / f"{mesh_attribute}.gii.gz"
            )

            if old_mesh_path.exists():
                old_surf = surface.load_surf_data(str(old_mesh_path))
                new_surf = surface.load_surf_data(str(new_mesh_path))

                assert np.array_equal(old_surf[0], new_surf[0])
                assert np.array_equal(old_surf[1], new_surf[1])
