import os
from nose.tools import assert_true
from nilearn.datasets import fetch_atlas_smith_2009
from nilearn_sandbox.plotting.pdf_plotting import plot_to_pdf
from tempfile import mkdtemp
from os.path import join

def test_plot_to_pdf():
    # Smoke test pdf plotting
    smith = fetch_atlas_smith_2009()
    img = smith.rsn10
    temp_dir = mkdtemp()
    file_path = join(temp_dir, 'output.pdf')
    plot_to_pdf(img, path=file_path)
    assert_true(os.path.exists(file_path))
    os.remove('output.pdf')
    os.rmdir(temp_dir)
