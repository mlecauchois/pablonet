FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Define working directory
WORKDIR /pablonet

# Remove existing SSH host keys
RUN rm -f /etc/ssh/ssh_host_*

# Install necessary packages
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends git wget curl bash libgl1 software-properties-common openssh-server nginx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Display Python version and create a Python virtual environment
RUN python3 --version && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN source venv/bin/activate && \
    pip install torch==2.1.0 torchvision==0.16.0 xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt && \
    pip install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4 --no-cache-dir && \
    pip install polygraphy==0.47.1 --extra-index-url https://pypi.ngc.nvidia.com && \
    pip install onnx-graphsurgeon==0.3.26 --extra-index-url https://pypi.ngc.nvidia.com

# Copy application files
COPY server.py .
COPY start.sh .

# Make the start script executable
RUN chmod +x start.sh

RUN source venv/bin/activate && pip list

# Set the start script as the default command
CMD ["./start.sh"]
