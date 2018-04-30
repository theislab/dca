from setuptools import setup

setup(
    name='DCA',
    version='0.1',
    description='Count autoencoder for scRNA-seq denoising',
    author='Gokcen Eraslan',
    author_email="gokcen.eraslan@gmail.com",
    packages=['DCA'],
    install_requires=['numpy>=1.7',
                      'keras>=2.0.8',
                      'h5py',
                      'six>=1.10.0',
                      'scikit-learn',
                      'scanpy',
                      'kopt',
                      'pandas' #for preprocessing
                      ],
    url='https://github.com/gokceneraslan/countae',
    entry_points={
        'console_scripts': [
            'dca = DCA.__main__:main'
    ]},
    license='Apache License 2.0',
    classifiers=['License :: OSI Approved :: Apache Software License',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Programming Language :: Python :: 3.5'],
)
