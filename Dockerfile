FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system deps
RUN apt-get update && apt-get install -y git wget unzip libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy your worker repo files (handler, configs, etc.)
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -r requirements.txt \
 && python3 -m pip cache purge

# Clone upstream 3DMM repo as third_party
RUN git clone https://github.com/ascust/3DMM-Fitting-Pytorch third_party/3DMM-Fitting-Pytorch

# Copy the rest of your worker (after requirements installed to avoid cache bust)
COPY . /app

EXPOSE 8000
CMD ["python", "-u", "handler.py"]
