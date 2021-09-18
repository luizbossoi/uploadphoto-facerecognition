# OpenCV Face Recognition

This script uses Flask + OpenCV to identify two photos and give the similarity percentage.
To run this using docker

    docker build . -t facerecog:latest
    docker run -p 3000:3000 facerecog:latest

* Server runs at http://localhost:3000


Flask Route Endpoints
    /front - html page to test the script
    /recognize - endpoint to upload photos and get comparison result


Only JPEG/JPG supported.