# AIDAR - Disaster Management

As part of the [TUM.ai Makeathon 2022](https://makeathon.tum-ai.com/), we created AIDAR. Checkout our corresponding repository [AIDAR-ui](https://github.com/jacob271/AIDAR-ui) to learn more about our product.

This repository is used to provide an api to identify destroyed areas in satellite images. Our work is based the repository [Destruction-Detection-in-Satellite-Imagery](https://github.com/usmanali414/Destruction-Detection-in-Satellite-Imagery) from @usmanali414. Thanks for providing this awesome model!

## Setup and Installation

Make sure to have `python3.8` or newer installed.

Install the requirements with `pip install -r requirements.txt`.

If you have trouble installing pydensecrf, check out [this discussion](https://github.com/lucasb-eyer/pydensecrf/issues/78).

Start the flask server with `uvicorn aidar:app`.

### Manually analysing an image

`python segmentation.py --image_path path/to/test_image.jpg --apply_crf yes`

