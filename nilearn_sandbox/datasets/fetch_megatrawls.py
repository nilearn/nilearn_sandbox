"""
Fetching Megatrawls NetMats data
"""
import os
import numpy as np

from sklearn.datasets.base import Bunch

from nilearn import datasets
from nilearn.datasets.utils import _get_dataset_dir


def fetch_megatrawl_netmats(data_dir=None, verbose=1):
    """ Load Network Matrices data from MegaTrawls release in HCP.

    Parameters
    ----------
    data_dir: string, default is None, optional
        Path of the data directory. Used to force data storage in a specified
        location.


    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like objet, attributes are:
        'netmats1': Subjects Measures Prediction, Full correlation matrices of
                    all dimensionalities.
        'netmats2': Subjects Measures Regression, Partial correlation matrices of
                    all dimensionalities.

    References
    ----------
    For more details:
    Stephen Smith et al, HCP beta-release of the Functional Connectivity MegaTrawl.
    April 2015 "HCP500-MegaTrawl" release.

    https://db.humanconnectome.org/megatrawl/
    """
    dataset_name = 'Megatrawls'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    n_dimensionalities = ['d25', 'd50', 'd100', 'd200', 'd300']
    timeseries_methods = ['ts2', 'ts3']
    n_combinations = len(n_dimensionalities) * len(timeseries_methods)

    NetMats1 = []
    NetMats2 = []
    NetMats1_csv = []
    NetMats2_csv = []

    # fetching all the networks matrices text file paths
    for each_n_dim in n_dimensionalities:
        filename = os.path.join('3T_Q1-Q6related468_MSMsulc_' + str(each_n_dim))
        for each_timeseries_method in timeseries_methods:
            csv_data_netmats1 = []
            csv_data_netmats2 = []
            path = os.path.join(data_dir, filename + '_' + str(each_timeseries_method))
            netmats1 = os.path.join(path, 'Znet1.txt')
            files_1 = np.genfromtxt(netmats1, delimiter=',', dtype=None)
            for file_1 in files_1:
                f_1 = file_1.split("  ")
                csv_data_netmats1.append(f_1)
            np.savetxt((os.path.join(path, 'Znet1.csv')),
                       csv_data_netmats1, delimiter=',', fmt="%s")
            netmats1_csv = (os.path.join(path, 'Znet1.csv'))

            netmats2 = os.path.join(path, 'Znet2.txt')
            files_2 = np.genfromtxt(netmats2, delimiter=',', dtype=None)
            for file_2 in files_2:
                f_2 = file_2.split("  ")
                csv_data_netmats2.append(f_2)
            np.savetxt((os.path.join(path, 'Znet2.csv')),
                       csv_data_netmats2, delimiter=',', fmt="%s")
            netmats2_csv = (os.path.join(path, 'Znet2.csv'))
            # Files
            NetMats1.append(netmats1)
            NetMats2.append(netmats2)
            NetMats1_csv.append(netmats1_csv)
            NetMats2_csv.append(netmats2_csv)

    return Bunch(full_correlation_files_txt=NetMats1,
                 partial_correlation_files_txt=NetMats2,
                 full_correlation_files_csv=NetMats1_csv,
                 partial_correlation_files_csv=NetMats2_csv)
