# Smart Object Detector

## From generic detection to personalized AI vision

### Project overview and contribution

This project implements a highly specialized Computer Vision solution using Transfer Learning on the YOLOv8 Nano architecture. The goal was to efficiently adapt a generic model (pre-trained on 80 classes) to detect a narrow set of custom, everyday objects (Water Bottle, Headphones, Pencil Case).

The primary technical contribution is:

Massive Performance Uplift: Achieving near-perfect accuracy on custom classes (99.50% mAP).

Advanced Temporal Feature: Integrating an Object Timer logic to measure the duration an object stays visible in the frame (as demonstrated below).

### Visual proof and feature showcase

This project implements a lightweight and highly accurate Computer Vision system, based on the YOLOv8 Nano architecture, specialized to detect three custom personal objects (Water Bottle, Headphones, Pencil Case) in real-time.

Core Goal: Demonstrate the effectiveness of Transfer Learning to specialize a generic model on a small, controlled dataset.

Unfortunately, I could not upload the images showing the results in this Readme, but you can see them by downloading the "Visual proof" document in this same repository.

### Key performance results (Test set evaluation)

The model was rigorously evaluated against an independent, unseen Test Set. The comparison validates the entire project hypothesis.

<img width="763" height="206" alt="image" src="https://github.com/user-attachments/assets/3ca25e22-cba6-48f5-896f-f6b4fe10483e" />

### Academic limitation 

It is important to point out that, due to time constraints, the model was trained exclusively on images taken under evening lighting conditions. This likely explains why the fine-tuned model performs so well. However, the original plan was to capture images at different times of the day with varying lighting conditions. This limitation confirms that the model is heavily overfitted to the training environment.

### How I did the project and how you can do it too !

#### Installation and setup

1). Clone the Repository

<img width="687" height="105" alt="image" src="https://github.com/user-attachments/assets/ec4cfdf5-ea5d-4ebb-92c0-f155e1448c9e" />


2). Configure environment

I highly recommend using a dedicated virtual environment (venv or Conda). All source files are located in the codes/ directory.

<img width="699" height="99" alt="image" src="https://github.com/user-attachments/assets/4b1374f3-b426-489f-984d-f5a54e1bb0e4" />


3). Install Dependencies

All required Python libraries (Ultralytics, OpenCV, PyTorch, etc.) are listed here :

<img width="705" height="58" alt="image" src="https://github.com/user-attachments/assets/526c6f7b-5183-426a-a298-3b9c17f9e093" />


#### Project steps (how I did the project) :

##### Phase I : Data & preprocessing

<img width="813" height="477" alt="image" src="https://github.com/user-attachments/assets/d6d44fd0-4696-4087-9471-18cf2bc7b79d" />

##### Phase II : Training & evaluation

<img width="806" height="320" alt="image" src="https://github.com/user-attachments/assets/440f4a12-ad78-44de-9344-fac4d83a95ef" />

##### Phase III : Added feature

<img width="810" height="121" alt="image" src="https://github.com/user-attachments/assets/47a6ba46-c8c4-4325-bfa4-f6cb7518dea3" />










