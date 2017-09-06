from setuptools import setup
from subprocess import check_output, CalledProcessError

try:
    num_gpus = len(check_output(['nvidia-smi', '--query-gpu=gpu_name',
                                 '--format=csv']).decode().strip().split('\n'))
    tf = 'tensorflow-gpu>=1.0.0' if num_gpus > 1 else 'tensorflow>=1.0.0'
except (CalledProcessError, FileNotFoundError, OSError):
    tf = 'tensorflow>=1.0.0'


setup(
    name='autoencoder',
    version='0.1',
    description='An autoencoder implementation',
    author='Gokcen Eraslan',
    author_email="goekcen.eraslan@helmholtz-muenchen.de",
    packages=['autoencoder'],
    install_requires=[tf,
                      'numpy>=1.7',
                      'keras>=2.0.8',
                      'six>=1.10.0',
                      'scikit-learn',
                      'zarr',
                      'pandas' #for preprocessing
                      ],
    url='https://github.com/gokceneraslan/autoencoder',
    entry_points={
        'console_scripts': [
            'autoencoder = autoencoder.__main__:main'
    ]},
    license='Apache License 2.0',
    classifiers=['License :: OSI Approved :: Apache Software License',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.5'],
)
