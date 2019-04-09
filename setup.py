from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='streamlines',
    version='0.0.0',
    packages=['streamlines'],
    scripts=['scripts/streamlines'],
    url='https://github.com/sdeslauriers/streamlines',
    license='GPL-3.0',
    author='Samuel Deslauriers-Gauthier',
    author_email='sam.deslauriers@gmail.com',
    description='A Python package to manipulate diffusion MRI streamlines',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['nibabel', 'numpy', 'scipy'],
    classifiers=(
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ),
)