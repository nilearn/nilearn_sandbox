from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn_sandbox.reporting.papaya import papaya_viewer


papaya_viewer('ho.html', fetch_atlas_harvard_oxford('cort-prob-2mm').maps)
