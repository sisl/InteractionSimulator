#
# File: setup.py
#
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="intersim",
    version="0.0.3",
    packages=find_packages(),
    description="Simulator environment for INTERACTION driving dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Arec Jamgochian, Etienne Buehrle, Johannes Fischer, Kunal Menda",
    author_email="jamgochian95@gmail.com",
    url="https://github.com/sisl/InteractionSimulator",
    install_requires=[
        "click",
        "joblib",
        "pandas",
        "matplotlib",
        "numpy",
        "scipy",
        "torch",
        "termcolor",
        "python_dateutil",
        "gym",
        "opencv-python",
        "celluloid",
        "future",
        "sklearn",
        "pyproj",
        "pytest",
        "shapely",
        "stable-baselines3",
        "jaxlib",
        "jax",
    ],    
    python_requires=">=3.6",
    test_requires=["pytest"],
    zip_safe=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],)