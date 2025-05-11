import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="range_ex",
    version="1.1.0",
    author="Niels Mündler, Raj Kiran P",
    author_email="n.muendler@posteo.de, rajkiranjp@gmail.com",
    description="Generate regex for numerical ranges",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nielstron/range_ex",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
