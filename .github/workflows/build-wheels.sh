#!/bin/bash
set -e -x

# Install a system package required by our library
yum install -y atlas-devel

# Install requirements
PYBIN=/opt/python/${PYABI}/bin
"${PYBIN}/pip" install -r /io/requirements-wheel.txt
"${PYBIN}/python" -c "import pyarrow; pyarrow.create_library_symlinks()"

# Compile wheels
cd /io
"${PYBIN}/python" setup.py bdist_wheel


# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    auditwheel repair "$whl" -w dist/
done

rm dist/*-linux*.whl
