# Real-Time Age & Gender Detection 🧠📸

A real-time computer vision application that detects faces from a live webcam feed and estimates **age range** and **gender** using pretrained deep convolutional neural networks — no training required, runs on CPU.

![Demo](demo.gif)
<!-- Replace demo.gif with your own screen recording (see "Adding a demo" below) -->

## How It Works

1. **Face Detection** — An OpenCV DNN-based SSD face detector (`opencv_face_detector`) locates all faces in each frame.
2. **Age & Gender Estimation** — Each detected face is cropped and passed through two separate pretrained CNNs:
   - An **age classifier** predicting one of 8 age brackets: `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, `(60-100)`
   - A **gender classifier** predicting `Male` or `Female`
3. **Live Overlay** — Results are drawn on the video feed in real time as bounding boxes with labels.

All inference runs through OpenCV's `dnn` module, so no GPU or deep learning framework installation (TensorFlow/PyTorch) is required.

## Requirements

- Python 3.8+
- A webcam

## Installation

```bash
git clone https://github.com/<your-username>/age-gender-detector.git
cd age-gender-detector
pip install -r requirements.txt
```

## Download the Pretrained Models

The two large model weight files (`age_net.caffemodel` and `gender_net.caffemodel`, ~44MB each) are not included in this repository. Download the following 6 files from the [LearnOpenCV AgeGender repository](https://github.com/spmallick/learnopencv/tree/master/AgeGender) and place them **in the same folder as `age_gender_detector.py`** (the project root):

| File | Description |
|---|---|
| `opencv_face_detector.pbtxt` | Face detector config |
| `opencv_face_detector_uint8.pb` | Face detector weights |
| `age_deploy.prototxt` | Age model config |
| `age_net.caffemodel` | Age model weights |
| `gender_deploy.prototxt` | Gender model config |
| `gender_net.caffemodel` | Gender model weights |

Your folder structure should look like:

```
age-gender-detector/
├── age_gender_detector.py
├── requirements.txt
├── README.md
├── opencv_face_detector.pbtxt
├── opencv_face_detector_uint8.pb
├── age_deploy.prototxt
├── age_net.caffemodel        (download separately, see above)
├── gender_deploy.prototxt
└── gender_net.caffemodel     (download separately, see above)
```

## Usage

```bash
python age_gender_detector.py
```

Press **q** to quit.

## Notes & Limitations

- Age is predicted as a **range**, not an exact number, since the model is trained on discrete brackets rather than continuous age values.
- Like any classifier, predictions aren't always perfectly accurate — this is a demo of applied deep learning, not a production-grade or clinically validated tool.
- Works best in good, even lighting with the face clearly visible to the camera.

## Adding a Demo GIF

To replace the placeholder above with your own demo:
1. Record a short screen capture while running the app.
2. Convert it to a GIF (e.g. with [ScreenToGif](https://www.screentogif.com/) on Windows, or `ffmpeg`).
3. Save it as `demo.gif` in the project root.

## License

This project is released under the [MIT License](LICENSE). The pretrained models are provided by the LearnOpenCV project — refer to their repository for licensing details.

## Acknowledgments

- Face/Age/Gender models originally from [LearnOpenCV](https://github.com/spmallick/learnopencv/tree/master/AgeGender), based on the work of Levi and Hassner (2015), *Age and Gender Classification Using Convolutional Neural Networks*.
