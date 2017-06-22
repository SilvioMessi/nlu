from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()
 
setup(
    name='nlu',
    packages=['nlu'],
    version='0.1',
    author='Silvio Messi',
    author_email='messisilvio@gmail.com',
    install_requires=required
)