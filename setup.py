from distutils.core import setup

setup(
    name='Pyeo',
    version='0.2.0',
    author='John Roberts',
    author_email='jfr10@le.ac.uk',
    packages=['pyeo', 'pyeo.tests'],
    url='http://pypi.python.org/pypi/Pyeo/',
    license='LICENSE',
    description='Modular processing chain from download to ard',
    long_description=open('README.md').read(),
    install_requires=[
        'pyshp >= 1.2.12',
        'sentinelsat >= 0.12.1'
    ],
)