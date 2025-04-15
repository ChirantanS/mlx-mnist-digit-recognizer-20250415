# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster
# (Choose a specific Python version close to what you used, e.g., 3.11, 3.10. Avoid 3.13 for now if dependencies might lag)

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed system dependencies (if any arise - common for some ML/CV libs, maybe not needed here)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local code directory contents into the container at /app
# This includes app.py, model.py, mnist_weights_transposed.npz, etc.
COPY . .

# Make port 8501 available to the world outside this container (Streamlit default port)
EXPOSE 8501

# Define environment variable needed by Streamlit (optional but good practice)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]