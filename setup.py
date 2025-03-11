from setuptools import find_packages, setup

setup(
    name='nycu-cv-hw1',
    packages=find_packages(exclude=[]),
    version='0.0.1',
    license="Proprietary",
    description='NYCU CV 2025 Spring - Homework 1',
    author='Enfu Liao',
    author_email='enfu.liao.cs10@nycu.edu.tw',
    long_description_content_type='text/markdown',
    url='https://github.com/lnfu/nycu-cv-hw1',
    keywords=[
        'ResNet',
    ],
    install_requires=[
        'torch==2.6.0',
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.12',
    ],
)
