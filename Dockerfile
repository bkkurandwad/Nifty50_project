# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code into the container
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Set the Streamlit command to run the app
CMD ["streamlit", "run", "UI.py"]