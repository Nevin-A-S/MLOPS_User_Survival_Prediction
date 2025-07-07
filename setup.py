from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="user-survival-prediction",
    version="0.1.0",
    author="Nevin A S",
    author_email="nevinajithkumar@gmail.com",
    packages=find_packages(),
    install_requires=requirements,
    )