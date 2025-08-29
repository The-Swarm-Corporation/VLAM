#!/usr/bin/env python3
"""Setup script for VLAM package."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="vlam",
    version="0.1.0",
    author="Kye Gomez",
    author_email="kye@swarms.world",
    description="Vision-Language-Action Model with Mamba State Space Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kyegomez/vlam",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "ruff",
            "mypy",
            "pytest",
            "pytest-benchmark",
            "bandit",
            "isort",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
        ],
    },
    entry_points={
        "console_scripts": [
            "vlam-demo=vlam.main:demo_main",
        ],
    },
    keywords=[
        "vision-language-action",
        "mamba",
        "state-space-models",
        "robotics",
        "multimodal",
        "ai",
        "machine-learning",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/kyegomez/vlam/issues",
        "Source": "https://github.com/kyegomez/vlam",
        "Documentation": "https://vlam.readthedocs.io/",
    },
    include_package_data=True,
    zip_safe=False,
)
