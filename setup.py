from setuptools import setup, find_packages # type: ignore
from typing import List

forbidden_line = "-e ."
def get_requirements(file_path: str)->List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

    if forbidden_line in requirements:
        requirements.remove(forbidden_line)

    return requirements


setup(
    name="ML_Project1",
    version="0.0.1",
    author="YASH",
    author_email="yashmalihan3@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)