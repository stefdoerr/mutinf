import setuptools
import unittest


if __name__ == '__main__':
    with open('README.md', 'r') as fh:
        long_description = fh.read()

    setuptools.setup(
        name='mutinf',
        version='0.0.1',
        author='Stefan Doerr',
        author_email='',
        description='MutInf',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/stefdoerr/mutinf/',
        classifiers=[
            'Programming Language :: Python :: 2.7',
            'Operating System :: POSIX :: Linux',
        ],

        packages=setuptools.find_packages(include=['mutinf*'], exclude=[]),

        install_requires=[
            'six==1.12.0',
            'numpy==1.22.0',
            'weave==0.17',
            'scipy==1.2.1',
            'biopython==1.73',
            'mdanalysis==0.19.2',
            'pymbar==3.0.3'
        ],

        entry_points = {
            'console_scripts': ['dihedral_mutent=mutinf.dihedral_mutent:main'],
        }
    )