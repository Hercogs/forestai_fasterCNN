FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install additional dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy application files
COPY . /app

# Configure the application
WORKDIR /app

# Install pip packages
RUN pip install -r requirements.txt

#ENTRYPOINT ["/bin/bash"]
#CMD ["bash"]

