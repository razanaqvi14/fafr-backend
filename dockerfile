FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
