from setuptools import setup, find_packages
import os

# Read version
def get_version():
    here = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(here, 'AgroGraphNet', '__version__.py')
    version_dict = {}
    with open(version_file) as f:
        exec(f.read(), version_dict)
    return version_dict['__version__']

# Read requirements
def get_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='agrographnet',
    version=get_version(),
    author='AgroGraphNet Team',
    author_email='contact@agrographnet.com',
    description='Graph Neural Networks for Agricultural Disease Prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/agrographnet',
    packages=find_packages(include=['AgroGraphNet', 'AgroGraphNet.*', 'agrographnet', 'agrographnet.*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.8',
    install_requires=get_requirements(),
    extras_require={
        'dev': ['pytest', 'pytest-cov', 'black', 'flake8', 'mypy'],
        'viz': ['plotly', 'dash'],
    },
    entry_points={
        'console_scripts': [
            'agrographnet=AgroGraphNet.cli.main:cli',
        ],
    },
    include_package_data=True,
    package_data={
        'agrographnet': [
            'templates/*',
            'templates/**/*',
        ],
    },
)