from setuptools import setup

setup(
    name='countae',
    version='0.2',
    description='A deep count autoencoder implementation for single cell RNA-seq data',
    author='Gokcen Eraslan',
    author_email="goekcen.eraslan@helmholtz-muenchen.de",
    packages=['countae'],
    install_requires=['numpy>=1.7',
                      #'pytorch', not on pypi yet
                      'h5py',
                      'six>=1.10.0',
                      'scikit-learn',
                      'zarr',
                      'pandas' #for preprocessing
                      ],
    url='https://github.com/gokceneraslan/autoencoder',
    entry_points={
        'console_scripts': [
            'countae = countae.__main__:main'
    ]},
    license='Apache License 2.0',
    classifiers=['License :: OSI Approved :: Apache Software License',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Programming Language :: Python :: 3.5'],
)
