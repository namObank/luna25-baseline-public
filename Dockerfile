# Base image
FROM python:3.10-slim

# # Base image Ubuntu
# FROM ubuntu:22.04

# # install tools
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         python3.10 \
#         python3.10-venv \
#         python3.10-dev \
#         python3-pip \
#         bash \
#         vim \
#         curl \
#         wget \
#         git \
#         build-essential \
#         libssl-dev \
#         libffi-dev \
#     && rm -rf /var/lib/apt/lists/*


# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

# Set working dir
WORKDIR .

# Copy code
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
