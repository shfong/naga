from setuptools import setup, find_packages

import re

# Get version number 
VERSIONFILE="nbgwas/version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

setup(
    name='naga-gwas',
    version=verstr,
    description='Network Assisted Genomic Association',
    url='https://github.com/shfong/naga',
    author='Samson Fong',
    author_email='shfong@ucsd.edu',
    license='MIT',
    long_description=readme + '\n\n' + history,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(exclude=['os', 're', 'time']),
    install_requires=[
            #Not sure why these packages were not automatically installed....
            'ijson',
            'requests', 
            'pytz>=2011k',
            'cycler>=0.10',
            'kiwisolver>=1.0.1',
            'pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1', 
            'pytz',

            #Main packages that are needed
            'networkx==1.11', #ndex2 requires networks 1.11
            'numpy',
            'matplotlib',
            'pandas>=0.23.0',
            'scipy',
            'seaborn', 
            'ndex2', 
            'python-igraph',
            'py2cytoscape',
            'coverage',
            'pytest',
            'mygene',
            'scikit-learn',
            'sphinx' # This should make sphinx-build 
                     # available in the virtual environment
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
