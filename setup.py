import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(name='sadptprj_riclyap_adi',
                 version='1.0.0',
                 description='Solve saddle-point problems as they occur' +
                 ' in simulations, model reduction, and optimal control' +
                 ' of incompressible flows.',
                 license="MIT",
                 long_description=long_description,
                 author='Jan Heiland',
                 author_email='jnhlnd@gmail.com',
                 url="https://github.com/highlando/sadptprj_riclyap_adi",
                 packages=['sadptprj_riclyap_adi'],  # same as name
                 install_requires=['numpy', 'scipy', 'krypy'],  # extrnl pckgs
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                     ]
                 )
