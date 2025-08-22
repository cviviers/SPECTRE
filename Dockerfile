FROM nvcr.io/nvidia/pytorch:25.04-py3

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install NVIDIA GDS user-space libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget gnupg \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    nvidia-gds \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code
COPY spectre /app/spectre
COPY experiments /app/experiments
COPY scripts /app/scripts

# Set the working directory
WORKDIR /app
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Set environment variables for GDS tuning
ENV CUFILE_ENV_PATH="/etc/cufile.json"
ENV LIBCUFILE_PATH="/usr/lib/x86_64-linux-gnu/libcufile.so"
# Optional: slightly better performance on NVMe-backed filesystems
ENV CUFILE_DISABLE_FS_POLLING=1