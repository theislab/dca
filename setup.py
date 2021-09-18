from setuptools import setup

setup(
    name='DCA',
    version='0.3.3',
    description='Count autoencoder for scRNA-seq denoising',
    author='Gokcen Eraslan',
    author_email="gokcen.eraslan@gmail.com",
    packages=['dca'],
    install_requires=['numpy>=1.7',
                      'keras>=2.4,<2.6',
                      'tensorflow>=2.0,<2.5',
                      'h5py',
                      'six>=1.10.0',
                      'scikit-learn',
                      'scanpy',
                      'kopt',
                      'pandas' #for preprocessing
                      ],
    url='https://github.com/theislab/dca',
    entry_points={
        'console_scripts': [
            'dca = dca.__main__:main'
    ]},
    license='Apache License 2.0',
    classifiers=['License :: OSI Approved :: Apache Software License',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Programming Language :: Python :: 3.5'],
)
