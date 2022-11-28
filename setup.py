from setuptools import setup

setup(
    name='DCA',
    version='0.3.4',
    description='Count autoencoder for scRNA-seq denoising',
    author='Gokcen Eraslan',
    author_email="gokcen.eraslan@gmail.com",
    packages=['dca'],
    install_requires=['numpy>=1.7',
                      'tensorflow>=2.0,<2.11,!=2.6',
                      'protobuf<=3.20',
                      'h5py',
                      'six>=1.10.0',
                      'scikit-learn',
                      'scanpy',
                      'kopt',
                      'pandas' #for preprocessing
                      ],
    extras_require={"test":["pytest"]},
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
