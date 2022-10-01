# AIDAR - Disaster Management

This is based on the work of ... 


## Setup and Installation

Since this model is based on `tensorflow 1.12.0`, you'll need `python3.6`.

Install the requirements with `pip install -r requirements.txt`.

Start the flask server with `uvicorn aidar:app`.

### Manually analysing an image

`python segmentation.py --image_path path/to/test_image.jpg --apply_crf yes`

