from setuptools import setup, find_packages

setup(
    name="active_learning_framework",
    version="1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
)
