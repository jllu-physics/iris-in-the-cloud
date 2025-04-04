# Use official TensorFlow CPU image as base
FROM tensorflow/tensorflow:2.18.0

# Environment Variable for Model Version, change accordingly
ENV MODEL_VERSION="v1"

# Environment Variable for Deployment Environment, change accordingly
ENV DEPLOY_ENVIRON="stage"

# Set working directory
WORKDIR .

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code into the container
COPY . .

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
