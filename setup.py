import setuptools

setuptools.setup(
    name='rlbox',
    version='0.0.1',
    author='Piotr Picheta',
    author_email='ocraftproject@gmail.com',
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    setup_requires=[
        'pytest-runner >= 4.2'
    ],
    tests_require=[
        'pytest == 4.0.2'
    ],
    install_requires=[
        'absl-py >= 0.7.0',
        'h5py >= 2.9.0',
        'matplotlib >= 3.0.2',
        'numba >= 0.42',
        'numpy >= 1.15',
        'pandas >= 0.24',
        'tables >= 3.4',
        'tqdm >= 4.31.1'
    ]
)
