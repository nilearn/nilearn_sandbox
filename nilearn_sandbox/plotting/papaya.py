import base64
import tempfile
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
    _, filename = tempfile.mkstemp(suffix='.nii.gz')
    f.to_filename(filename)
    with open(filename, 'rb') as f:
        b64 = base64.b64encode(f.read())
    os.unlink(filename)
    return b64.decode('utf-8')


def papaya_viewer(maps_niimg, output_file=None):
    # The template is tempita compliant but I don't include it just for a
    # simple replace. If it becomes a dependency, use it here.
    # import tempita

    open_in_browser = (output_file is None)
    if open_in_browser:
        _, output_file = tempfile.mkstemp(suffix='.html')

    body = ""
    package_directory = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(package_directory, "data", "papaya")

    with open(os.path.join(data, 'template.html'), 'rb') as f:
        template = f.read().decode('utf-8')
    javascript = u''
    javascript += 'var maps = "'
    javascript += _get_64(maps_niimg)
    javascript += '";\n'
    javascript += 'var template = "'
    javascript += _get_64(os.path.join(data, 'template.nii.gz'))
    javascript += '";\n'

    with open(os.path.join(data, 'papaya.js'), 'rb') as f:
        javascript += f.read().decode('utf-8')
    with open(os.path.join(data, 'papaya.css'), 'rb') as f:
        css = f.read().decode('utf-8')

    text = template.replace('{{css}}', css)
    text = text.replace('{{javascript}}', javascript)
    text = text.replace('{{body}}', body)

    with open(output_file, 'wb') as m:
        m.write(text.encode('utf-8'))

    if open_in_browser:
        from nilearn._utils.compat import _urllib
        import webbrowser

        webbrowser.open(_urllib.request.pathname2url(output_file))
