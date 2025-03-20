#!/usr/bin/env bash

set -e

if [ $# -eq 0 ]; then
    echo "please write the type of release you would like to perform (major, minor, patch, or a specific version)"
    exit 1
fi

poetry version $1
version=$(poetry version --short)

# Update __version__.py file in place
#sed -i '' 's/^__version__ = .*/__version__ = "'"$version"'"/' portfolyo/__version__.py
echo "__version__ = \"$version\"" > portfolyo/__version__.py



