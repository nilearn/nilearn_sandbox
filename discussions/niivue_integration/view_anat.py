# %%
import base64
import json
import string

from pathlib import Path

from nilearn import datasets
from nilearn.plotting import html_document
from nilearn.plotting.img_plotting import _MNI152Template

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
MNI152TEMPLATE = _MNI152Template()


# %%
def view_anat(
    anat_img,
    # cut_coords=None, # Seems to be missing from niivue
    threshold=None,
    draw_cross=True,
    # cmap=plt.cm.gray,
    # colorbar=False,
):
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
            loadingText: "Loading...",
            backColor: [1, 1, 1, 1],
            show3Dcrosshair: $draw_cross,
            onLocationChange: handleLocationChange,
        });

        nv.setRadiologicalConvention(false);
        nv.attachTo("gl1");
        nv.setSliceType(nv.sliceTypeMultiplanar);
        nv.setSliceMM(false);
        nv.opts.isColorbar = true;

        let anat = niivue.NVImage.loadFromBase64({
            name: "bg_map.nii.gz",
            base64: "$anat_img",
            cal_min: $threshold,
            cal_max: 10000,
        });
        nv.addVolume(anat);

        nv.volumes[0].cal_minNeg = -10000;
        nv.volumes[0].cal_maxNeg = -$threshold;

        nv.opts.multiplanarForceRender = true;
        nv.setInterpolation(true);
        nv.updateGLVolume();
        </script>
    </body>
    </html>
    """
    )

    encoded = {}

    encoded["anat_img"] = base64.b64encode(anat_img.read_bytes()).decode(
        "UTF-8"
    )

    encoded["draw_cross"] = "true" if draw_cross else "false"

    encoded["threshold"] = "null" if threshold is None else threshold

    display = html_document.HTMLDocument(template.safe_substitute(encoded))

    return display


# %%
display = view_anat(
    localizer_anat_filename,
    threshold=None,
    draw_cross=True,
)

display.save_as_html("/home/alexis/singbrain/data/tmp/niivue_plot_anat.html")
