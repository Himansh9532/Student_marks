from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."

# Function to read and return requirements from a text file
def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]  # Corrected replace function
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="Student_Marks",  # Replace with your package name
    version="0.1.0",  # Initial release version
    author="Himanshu",
    author_email="himanshugupta95326@gmail.com",
    description="A brief description of your package",
    url="https://github.com/Himansh9532/Student_marks.git",  # Project's homepage URL
    packages=find_packages(),  # Automatically find all packages
    install_requires=get_requirements("requirements.txt")  # Read requirements from a file
)
