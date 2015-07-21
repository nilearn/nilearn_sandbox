#! /usr/bin/env python

descr = """Playground for nilearn compatible features."""

import os

from setuptools import setup, find_packages


# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DISTNAME = 'nilearn-sandbox'
DESCRIPTION = 'Playground for nilearn compatible features.'
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Alexandre Abraham'
MAINTAINER_EMAIL = 'abraham.alexandre@gmail.com'
URL = 'https://github.com/AlexandreAbraham/nilearn-sandbox'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/AlexandreAbraham/nilearn-sandbox'
VERSION = '0.1a'


if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 2',
              'Programming Language :: Python :: 2.6',
              'Programming Language :: Python :: 2.7',
          ],
          packages=find_packages())
