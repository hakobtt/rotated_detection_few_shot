from distutils.core import setup
import pathlib
import pkg_resources
import setuptools

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setuptools.setup(
    name='Hakob Redet',
    version='0.0',
    description='SSOD for rotated objects',
    author='Hakob Kirakosyan',
    author_email='hakobdilif@gmai.com',
    packages=['ssod', ],
    install_requires=install_requires,
)
