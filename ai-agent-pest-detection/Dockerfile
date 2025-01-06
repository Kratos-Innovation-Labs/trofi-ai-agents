FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code into the container
COPY . /app

# Expose the port Flask runs on
EXPOSE 5000

# Run the Flask app
CMD ["python", "./app/app.py"]
