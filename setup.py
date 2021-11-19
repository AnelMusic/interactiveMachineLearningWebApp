"""
Created on Tue Aug 24 00:46:26 2021
@author: anelmusic
"""


from pathlib import Path

from setuptools import setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

dev_packages = [
    "black==20.8b1",
    "flake8==3.8.3",
    "isort==5.5.3",
    "pre-commit==2.11.1",
]

setup(
    name="Interactive ML Webapp using Streamlit",
    version="0.1",
    license="MIT",
    description="Tag suggestions for projects on Made With ML.",
    python_requires=">=3.6",
    install_requires=[required_packages],
    extras_require={"dev": dev_packages},
    entry_points={
        "console_scripts": ["mlapp = app.wrapped_cli_tool:main"],
    },
)
