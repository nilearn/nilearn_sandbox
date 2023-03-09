# %%
import string
import base64
import pathlib

import nibabel as nib
import numpy as np

from nilearn import datasets, surface
from nilearn.plotting import html_document

# %%
fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage7")

# %%
# Create curv sign map
curv_sign_left = (np.sign(surface.load_surf_data(fsaverage.curv_left)) + 1) / 2
gifti_image = nib.gifti.GiftiImage()
gifti_image.add_gifti_data_array(
    nib.gifti.GiftiDataArray(curv_sign_left, "NIFTI_INTENT_NONE")
)
nib.save(gifti_image, "/home/alexis/singbrain/data/tmp/curv_sign_left.gii")


# %%
motor_images = datasets.fetch_neurovault_motor_task()
stat_img = motor_images.images[0]
surface_map = surface.vol_to_surf(stat_img, fsaverage.pial_left)

# %%
surface_map_path = "/home/alexis/singbrain/data/tmp/surface_map.gii"

img = nib.gifti.gifti.GiftiImage()
img.add_gifti_data_array(
    nib.gifti.gifti.GiftiDataArray(
        surface_map,
        intent="NIFTI_INTENT_ZSCORE",
    )
)
nib.save(img, surface_map_path)

# %%
# <!-- url: "http://0.0.0.0:8000/surface_map.gii", -->
template = string.Template(
    """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>NiiVue</title>
    <link rel="stylesheet" href="https://niivue.github.io/niivue/features/niivue.css">
  </head>
  <body>
    <canvas id="gl"></canvas>
  </body>
    <script src="https://niivue.github.io/niivue/features/niivue.umd.js"></script>
    <script type="module" async>
      var nv = new niivue.Niivue();
      nv.attachTo('gl');
      let layers = [
        {
            name: "bg_map.gii",
            useNegativeCmap: true,
            opacity: 0.7,
            colorMap: "gray",
            base64: "$bg_map",
        },
        {
            name: "surface_map.gii",
            useNegativeCmap: true,
            opacity: 0.7,
            cal_min: $threshold,
            base64: "$surf_map",
        },
      ];
      let m = await niivue.NVMesh.loadFromBase64({
        gl: nv.gl,
        name: "pial_left.gii",
        layers: layers,
        base64:"$surf_mesh"
      });
      nv.addMesh(m);
    </script>
</html>
"""
)

# %%
encoded = {}

# %%
encoded["surf_mesh"] = base64.b64encode(
    pathlib.Path("/home/alexis/singbrain/data/tmp/pial_left.gii").read_bytes()
).decode("UTF-8")

# %%
encoded["surf_map"] = base64.b64encode(
    pathlib.Path(surface_map_path).read_bytes()
).decode("UTF-8")

# %%
encoded["threshold"] = 3

# %%
encoded["bg_map"] = base64.b64encode(
    pathlib.Path("/home/alexis/singbrain/data/tmp/curv_sign_left.gii").read_bytes()
).decode("UTF-8")

# %%
display = html_document.HTMLDocument(template.safe_substitute(encoded))

# save for later:
display.save_as_html("/home/alexis/singbrain/data/tmp/niivue_plot.html")
