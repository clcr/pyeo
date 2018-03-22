from distutils.core import setup

setup(
    name='Pyeo',
    version='0.1.0',
    author='John Roberts',
    author_email='jfr10@le.ac.uk',
    packages=['pyeo', 'pyeo.tests'],
    url='http://pypi.python.org/pypi/Pyeo/',
    license='LICENSE',
    description='Modular processing chain from download to ard',
    long_description=open('README.md').read(),
    install_requires=[
        'Shapely >= 1.6.2',
        'sentinelsat >= 0.12.1'
    ],
)