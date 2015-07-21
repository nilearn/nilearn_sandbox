import os
import cookielib
import getpass

from sklearn.datasets.base import Bunch

from nilearn import datasets
from nilearn._utils.compat import _urllib

from ._utils.compat import _input


def _get_nitrc_auth_cookie(login=None):
    if login is None:
        login = _input('login: ')
    passwd = getpass.getpass('password: ')

    login_url = "https://www.nitrc.org/account/login.php"

    # First, we login to NITRC
    cp = _urllib.request.HTTPCookieProcessor(cookielib.CookieJar())
    opener = _urllib.request.build_opener(cp)
    login_data = _urllib.parse.urlencode(
        {'form_loginname': login, 'form_pw': passwd})
    opener.open(login_url, login_data)

    return cp


def fetch_COBRE(data_dir=None, verbose=0):

    dataset_name = 'cobre'
    data_dir = datasets._get_dataset_dir(dataset_name, data_dir=data_dir,
                                         verbose=verbose)

    ids = ['0040%03d' % i for i in range(148)]
    cookie = _get_nitrc_auth_cookie()
    data_url = ('http://www.nitrc.org/frs/downloadlink.php/5066/'
                '?i_agree=1&download_now=1')
    anats = [os.path.join('COBRE', str(id_), 'session_1', 'anat_1',
                          'mprage.nii.gz') for id_ in ids]
    rests = [os.path.join('COBRE', str(id_), 'session_1', 'rest_1',
                          'rest.nii.gz') for id_ in ids]
    files_anats = [(path, data_url, dict(handlers=[cookie], uncompress=True))
                    for path in anats]
    files_rests = [(path, data_url, dict(handlers=[cookie], uncompress=True))
                    for path in rests]
    csvs = [
        ('COBRE_parameters_rest.csv',
         'http://www.nitrc.org/frs/downloadlink.php/5068/'
         '?i_agree=1&download_now=1',
         dict(handlers=[cookie], move='COBRE_parameters_rest.csv')),
        ('COBRE_parameters_mprage.csv',
         'http://www.nitrc.org/frs/downloadlink.php/5067/'
         '?i_agree=1&download_now=1',
         dict(handlers=[cookie], move='COBRE_parameters_mprage.csv')),
        ('COBRE_phenotypic_data.csv',
         'http://www.nitrc.org/frs/downloadlink.php/5071/'
         '?i_agree=1&download_now=1',
         dict(handlers=[cookie], move='COBRE_phenotypic_data.csv')),
    ]

    csvs = datasets._fetch_files(data_dir, csvs, verbose=verbose)
    files_anats = datasets._fetch_files(data_dir, files_anats, verbose=verbose)
    files_rests = datasets._fetch_files(data_dir, files_rests, verbose=verbose)

    return Bunch(anatomical=files_anats, functional=files_rests,
                 functional_parameters=csvs[0],
                 anatomical_parameters=csvs[1],
                 phenotypic=csvs[2])
