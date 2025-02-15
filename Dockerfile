# Use an official lightweight Python image as a base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install Node.js, npm, and other dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y \
    nodejs \
    tesseract-ocr \
    tesseract-ocr-eng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify installations and set Tesseract data path
RUN node -v && npm -v && tesseract --version
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port Flask runs on
EXPOSE 8000

# Set environment variables for production
ENV FLASK_APP=main.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "main.py"]