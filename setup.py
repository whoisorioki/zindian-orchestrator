from setuptools import find_packages, setup


setup(
    name="zindian",
    version="0.1.0",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "google-genai",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "tabula=tabula.__main__:main",
            "zindian-cli=zindian.cli:main",
        ]
    },
)
