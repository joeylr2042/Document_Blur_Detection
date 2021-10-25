# Document Blur Detection

For general blurred image, using the variance of Laplacian operator is a good solution. But as for the blur detection of documents, especially for document images with blurred text, text detection should be used to detect blurred text area.

This package mainly depends on opencv and paddle, to install them with requirements.txt,
```python
pip install -r requirements
```

Inference model of PaddleOCR is used to detect text location. You can download the inference model with https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar. The text detection code in this project refers to the PaddleOCR project. If you want to get more information about PaddleOCR, you can go to https://github.com/PaddlePaddle/PaddleOCR to check it out.

To run `main.py`, use the following command.
```python
python ./main.py --image './text_blur.jpg' --thresh_v 300 --thresh_d 0.7
```
If you would like to blur document images, you can run `blur_ops.py` to simulate motion blur and Gaussian blur. Use the following command.
```python
python blur_ops.py --image_path './bean-license.png' --output_path './gaussian_blur.jpg' --blur_type 'gaussian blur'/'motion blur'
```

