# %%
import base64
import string

from pathlib import Path

import nibabel as nib
import numpy as np

from nilearn import datasets, image, surface
from nilearn.plotting import html_document

# %%
output_path = Path("/home/alexis/singbrain/data/tmp")

# %%
localizer_dataset = datasets.fetch_localizer_button_task(legacy_format=False)
# Contrast map of motor task
localizer_tmap_filename = Path(localizer_dataset.tmap)
# Subject specific anatomical image
localizer_anat_filename = Path(localizer_dataset.anat)
# localizer_anat_filename = Path(datasets.MNI152_FILE_PATH)
localizer_anat_filename
localizer_tmap_filename

# %%
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
    <noscript>
      <strong>niivue requires JavaScript.</strong>
    </noscript>
    <main>
      <canvas id="gl1"></canvas>
    </main>
    <footer id="location">&nbsp;</footer>
    <script src="https://niivue.github.io/niivue/features/niivue.umd.js"></script>
    <script type="module" async>
      function handleLocationChange(data) {
        document.getElementById("location").innerHTML =
          "&nbsp;&nbsp;" + data.string;
      }

      var nv = new niivue.Niivue({
        loadingText: "there are no images",
        backColor: [1, 1, 1, 1],
        show3Dcrosshair: true,
        onLocationChange: handleLocationChange,
      });

      nv.setRadiologicalConvention(false);
      nv.attachTo("gl1");
      nv.setSliceType(nv.sliceTypeMultiplanar);
      nv.setSliceMM(false);
      nv.opts.isColorbar = true;

      let bg = niivue.NVImage.loadFromBase64({
        name: "bg_map.nii.gz",
        base64: "$bg_img",
        colorbarVisible: false,
      });
      nv.addVolume(bg);

      nv.volumes[0].colorbarVisible = false;

      let v = niivue.NVImage.loadFromBase64({
        name: "stat_map_img.nii.gz",
        base64: "$stat_map_img",
        colorMap: "warm",
        cal_min: $threshold,
        cal_max: 6,
      });
      nv.addVolume(v);

      nv.volumes[1].colorMapNegative = "winter";
      nv.volumes[1].alphaThreshold = true;
      nv.volumes[1].cal_minNeg = -6;
      nv.volumes[1].cal_maxNeg = -$threshold;

      nv.opts.multiplanarForceRender = true;
      nv.setInterpolation(true);
      nv.updateGLVolume();
    </script>
  </body>
</html>
"""
)

# %%
encoded = {}

# %%
encoded["bg_img"] = base64.b64encode(
    localizer_anat_filename.read_bytes()
).decode("UTF-8")

# %%
encoded["stat_map_img"] = base64.b64encode(
    localizer_tmap_filename.read_bytes()
).decode("UTF-8")

# %%
encoded["threshold"] = 3

# %%
display = html_document.HTMLDocument(template.safe_substitute(encoded))

# save for later:
display.save_as_html(
    "/home/alexis/singbrain/data/tmp/niivue_plot_surf_map.html"
)

# %%
