from setuptools import setup, find_packages
from typing import List


HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
  
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        # Remove newline characters
        requirements = [req.replace("\n", "") for req in requirements]

        # Remove editable install flag if present
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="Machine_Learning_Project",
    version="0.0.1",
    author="Ashmit Sharma",
    author_email="sashmit972@gmail.com",

    packages=find_packages(),

    install_requires=get_requirements("requirements.txt"),

    description="An End-to-End Machine Learning Project",

    long_description="""
    This project demonstrates the complete lifecycle
    of an End-to-End Machine Learning application,
    including data ingestion, validation,
    transformation, model training, evaluation,
    and deployment.
    """,

    python_requires=">=3.10",

    license="MIT",

    keywords=[
        "Machine Learning",
        "Data Science",
        "Python",
        "Flask",
        "ML Pipeline",
        "Deployment"
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)