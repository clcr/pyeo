# pyeo
Python Earth Observation processing chain

This is designed to provide a set of portable, extensible and modular Python scripts for earth observation in
CLCR.

## To install

With Git installed, `cd` to an install location then run the following lines

```bash
git clone https://github.com/clcr/pyeo.git
cd pyeo
python setup.py install -user
```

If you want to edit this code, run

```bash
python setup.py develop
```

instead of install

To verify the installation, open a Python prompt and type

```python
>>> import pyeo.core
```

You should get no errors.


Full documentation at https://clcr.github.io/pyeo/build/html/index.html
