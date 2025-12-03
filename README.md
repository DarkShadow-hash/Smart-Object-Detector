# Smart Object Detector

## Two projects, one mission: From generic detection to personalized AI vision

### Project Overview and Contribution

This project implements a highly specialized Computer Vision solution using Transfer Learning on the YOLOv8 Nano architecture. The goal was to efficiently adapt a generic model (pre-trained on 80 classes) to detect a narrow set of custom, everyday objects (Water Bottle, Headphones, Pencil Case).

The primary technical contribution is:

Massive Performance Uplift: Achieving near-perfect accuracy on custom classes (99.50% mAP).

Advanced Temporal Feature: Integrating an Object Timer logic to measure the duration an object stays visible in the frame (as demonstrated below).

### Visual Proof and Feature Showcase

This project implements a lightweight and highly accurate Computer Vision system, based on the YOLOv8 Nano architecture, specialized to detect three custom personal objects (Water Bottle, Headphones, Pencil Case) in real-time.

Core Goal: Demonstrate the effectiveness of Transfer Learning to specialize a generic model on a small, controlled dataset.

####                                             Key feature                                            Visual proof



                                               Before fine-tuning                                  ![image](https://github.com/user-attachments/assets/ba184727-da49-48a9-929e-22cb164b4785)






                                               
                                               After fine-tuning                                                    

### Key Performance Results (Test Set Evaluation)

The model was rigorously evaluated against an independent, unseen Test Set. The comparison validates the entire project hypothesis.
<img width="763" height="206" alt="image" src="https://github.com/user-attachments/assets/3ca25e22-cba6-48f5-896f-f6b4fe10483e" />

                                             


### Project steps :



