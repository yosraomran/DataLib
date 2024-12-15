from setuptools import setup, find_packages

setup(
    name="datalib",  
    version="0.1.0",  
    description="A library for data processing and analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Yosra Omrane",
    url="https://github.com/yosraomran/DataLib", 
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Minimum Python version
    install_requires=[
       "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "pytest",
    "twine",
    "sphinx"
    ],
)
