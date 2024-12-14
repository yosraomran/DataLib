from setuptools import setup, find_packages

setup(
    name="datalib",
    version="1.0.0",
    description="A Python library for data manipulation and ML",
    author="Yosra Omrane",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "pytest"
    ],
)
