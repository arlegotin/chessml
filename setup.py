from setuptools import setup, find_packages

setup(
    name="chessml",
    version="0.1.0",
    packages=find_packages(),
    author="Artem Legotin",
    author_email="arlegotin@gmail.com",
    description="A Python package for advanced chess analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/arlegotin/chessml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
) 