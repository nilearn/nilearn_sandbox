import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn.plotting import plot_stat_map, plot_prob_atlas
from nilearn._utils import check_niimg_4d
from nilearn.image import index_img


def plot_to_pdf(img, path='output.pdf', vmax=None):
    """Creates a pdf from a 4D nii-image like

    Parameters
    ----------
    img: nii-like image,
        Image to dump as pdf

    path: str,
        Path of the output pdf

    vmax: float or 'auto' or None,
        vmax to use in plot_stat_map. 'auto' will compute it magically
    """
    a4_size = (8.27, 11.69)
    img = check_niimg_4d(img)
    n_components = img.shape[3]

    if vmax == 'auto':
        vmax = np.max(np.abs(img.get_data()), axis=3)
        vmax[vmax >= 0.1] = 0
        vmax = np.max(vmax)
    elif vmax is not None:
        vmax = float(vmax)
    with PdfPages(path) as pdf:
        for i in range(-1, n_components, 5):
            fig, axes = plt.subplots(5, 1, figsize=a4_size, squeeze=False)
            axes = axes.reshape(-1)
            for j, ax in enumerate(axes):
                if i + j < 0:
                    plot_prob_atlas(img, axes=ax)
                elif j + i < n_components:
                    plot_stat_map(index_img(img, j + i), axes=ax, vmax=vmax)
                else:
                    ax.axis('off')
            pdf.savefig(fig)
            plt.close()
