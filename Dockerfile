# Use a base Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r Requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords punkt

# Default command
CMD ["python", "DevOps Project.py"]
