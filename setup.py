from setuptools import setup, find_packages

setup(
    name="zindian",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "google-genai",
        "requests",
    ],
)
