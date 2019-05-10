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
    install_requires=[
        'sentinelsat>=0.12.1',
        'sentinelhub>=2.4.2',
        'pytest>=3.5.0',
        'pygdal>=2.3.0',
        'numpy>=1.15.4',
        'scikit-learn==0.19.2',
        'scikit-image>=0.14.0'
        'scipy>=1.1.0',
        'joblib>=0.12.5',
        'requests>=12.20.1',
        'tenacity>=5.0.2',
        'pytest>=4.0.0'
    ],
)
