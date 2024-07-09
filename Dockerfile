FROM python:3.12

WORKDIR /app

# Copy the files
COPY pyproject.toml ./
COPY pycats ./pycats
COPY tests ./tests

# Install build tool
RUN pip install --upgrade pip setuptools pytest

# Install the package
RUN pip install .

# Run the tests
CMD ["pytest", "tests"]
