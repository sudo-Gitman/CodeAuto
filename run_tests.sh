#!/bin/bash

# Build test image
docker build -t autogpt-test -f Dockerfile.test .

# Run tests
docker run --rm \
    -v $(pwd)/tests:/app/tests \
    -v $(pwd)/logs:/app/logs \
    autogpt-test python -m pytest tests/ -v
