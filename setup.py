from setuptools import setup, find_packages

setup(
    name='lrf',
    version='0.1.0',
    description='Learnable Response Function (LRF) model for rainfall-derived inflow and infiltration (RDII)',
    author='Your Name',
    author_email='jingyuzoege@gmail.com',
    url='https://github.com//JingyuZoe/lrf',  # Replace with actual GitHub repo if available
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)