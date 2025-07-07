"""Setup script for spider locomotion environment."""

from setuptools import setup, find_packages

setup(
    name="spider_locomotion_isaaclab",
    version="0.1.0",
    description="Spider robot locomotion environment for Isaac Lab",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
    ],
    python_requires=">=3.10",
)
