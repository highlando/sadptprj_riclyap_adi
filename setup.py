from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='sadptprj_riclyap_adi',
      version='v1.1',
      description='A Scipy-Fenics interface for incompressible Navier-Stokes',
      license="GPLv3",
      long_description=long_description,
      author='Jan Heiland',
      author_email='jnhlnd@gmail.com',
      url="https://github.com/highlando/sadptprj_riclyap_adi",
      packages=['sadptprj_riclyap_adi'],  # same as name
      install_requires=['numpy', 'scipy']  # external packages as dependencies
      )
