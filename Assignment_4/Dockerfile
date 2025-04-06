FROM python:3.12.3-slim-bullseye

# Set working directory
WORKDIR /app

# Copy files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader punkt stopwords wordnet punkt_tab

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]