FROM python:3.9.13
# Set the working directory within the container
WORKDIR /app

# Copy your application requirements file to the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code to the container
COPY . /app
# ENTRYPOINT ["python"]
# Specify the command to run your application
# CMD ["app.py"]
EXPOSE 8081
CMD ["python", "app.py", "--host=0.0.0.0"]