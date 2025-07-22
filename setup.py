"""
Setup script for noise filtering project
"""

from setuptools import setup, find_packages
import sys

if sys.version_info < (3, 11):
    sys.exit('This project requires Python 3.11 or higher')

setup(
    name="noise-filtering-project",
    version="1.0.0",
    description="Environment-aware noise filtering for online meetings",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="DSP Group 3",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "matplotlib>=3.8.0",
        "pydub>=0.25.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
        'audio': [
            'sounddevice>=0.4.0',
            'plotly>=5.0.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'noise-filter=main:main',
            'noise-filter-demo=app:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    include_package_data=True,
    package_data={
        'templates': ['*.html'],
        'static': ['css/*.css', 'js/*.js'],
    },
)