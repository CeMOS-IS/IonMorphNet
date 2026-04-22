from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="ionmorphnet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.9',
    author="Niels Nawrot",
    author_email="n-nawrot@hotmail.com",
    description="Mass spectrometry imaging analysis and ion image classification tools",
    url="",
    include_package_data=True,
    package_data={
        'ionmorphnet': ['configs/*.yaml'],
    },
)
