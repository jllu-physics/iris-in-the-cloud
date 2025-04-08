# Use official TensorFlow CPU image as base
FROM tensorflow/tensorflow:2.18.0

# Environment Variables are now consistently specified in .env

# Set working directory
WORKDIR .

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code into the container
COPY . .

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
