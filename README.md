# Fish Counter Web Application

A web application that allows users to upload images of fish and uses a YOLO model to detect and count the fish in the images.

## Features

- Upload images (JPEG, PNG) containing fish
- Automatic detection and counting of fish using a pre-trained YOLO model
- Display of both original and processed images with detected fish
- Real-time image preview before upload
- User registration and authentication
- History tracking of analyses

## Requirements

- Python 3.11 or higher
- Flask 3.1.0 or higher
- Ultralytics 8.3.128 or higher
- OpenCV
- NumPy
- MongoDB

## Installation

1. Clone this repository
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Make sure you have MongoDB running locally or update the connection string in the code

## Usage

1. Run the application:
```
python app.py
```

2. Open a web browser and navigate to http://127.0.0.1:5000

3. Upload an image containing fish and click "Detect Fish"

4. View the detection results, including the count of fish detected

## Model Information

This application uses a YOLO model that was trained to detect fish. The model file is located in the `models` directory.

## Deployment

This application can be deployed using:

- **Render** for the backend: Handles Python, Flask and the YOLO model
- **Vercel** for the frontend: Serves the web interface

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

### Environment Variables

When deploying, you'll need to set the following environment variables:

- `MONGODB_URI`: Connection string for your MongoDB database
- `FLASK_SECRET_KEY`: Secret key for Flask session encryption
- `MODEL_URL`: URL to your YOLO model file in cloud storage (for cloud deployments)

A sample `env.example` file is included to help you set up your environment.

## License

[MIT License](LICENSE)
