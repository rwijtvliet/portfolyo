from setuptools import find_packages, setup

import versioneer

setup(
    name="portfolyo",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Ruud Wijtvliet",
    zip_safe=False,
    packages=find_packages(exclude=["tests"]),
    description="Analysing and manipulating timeseries related to power and gas offtake portfolios.",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    python_requires=">=3.8",
    install_requires=[line.strip() for line in open("requirements.txt", "r")],
    # package_data is data that is deployed within the python package on the
    # user's system. setuptools will get whatever is listed in MANIFEST.in
    include_package_data=True,
)
