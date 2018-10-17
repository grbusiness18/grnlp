import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="grnlp",
    version="0.0.1",
    author="Gokulraj Ramdass",
    author_email="gokulraj.ramdass@sap.com",
    description="GRNLP for Text Classification",
    long_description="GRNLP for Text Classification",
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)