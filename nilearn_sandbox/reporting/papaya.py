import base64
import nibabel
import numpy as np
import os
from nilearn._utils import check_niimg


def _get_64(niimg):
    f = check_niimg(niimg)
    data = f.get_data().astype(float)
    data = data + data.min()
    data = data / data.max() * 254.
    data = data.astype(np.uint8)
    f = nibabel.Nifti1Image(data, f.get_affine())
    f.to_filename('_maps.nii.gz')
    with open('_maps.nii.gz', 'br') as f:
        b64 = base64.b64encode(f.read())
    os.unlink('_maps.nii.gz')
    return b64.decode('utf-8')


def papaya_viewer(html_path, maps_niimg):
    import tempita

    body = ""
    package_directory = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(package_directory, "data", "papaya")

    with open(os.path.join(data, 'template.html'), 'r') as f:
        template = f.read()
    tmpl = tempita.Template(template)
    javascript = ''
    javascript += 'var maps = "'
    javascript += _get_64(maps_niimg)
    javascript += '";\n'
    javascript += 'var template = "'
    javascript += _get_64(os.path.join(data, 'template.nii.gz'))
    javascript += '";\n'
    with open(os.path.join(data, 'papaya.js'), 'r') as f:
        javascript += f.read()
    with open(os.path.join(data, 'papaya.css'), 'r') as f:
        css = f.read()
    text = tmpl.substitute(locals())
    with open(html_path, 'w') as m:
        m.write(text)
