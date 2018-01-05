from setuptools import setup

setup(name='bath',
      version='0.1',
      description='preservation draft',
      packages=['bath'],
      install_requires=[
         'numpy', 'keras',
      ],
      zip_safe=False)