from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn_sandbox.plotting.papaya import papaya_viewer


papaya_viewer(fetch_atlas_harvard_oxford('cort-prob-2mm').maps)
