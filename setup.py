from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="greedy-mixed-graph",
    version="0.1.0",
    author="wjmoss",
    author_email="wujun3544@gmail.com",
    description="tbd",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
        ],
    # packages=setuptools.find_packages(),
    py_modules=[
        "greedy_mixed_graph.generate",
        "greedy_mixed_graph.greedysearch",
        "greedy_mixed_graph.plot"
        "greedy_mixed_graph.ricf"
        ]
    )