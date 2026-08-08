# **Final Project** 

### **Object Detection Application** 

## **1 Overview** 

- Object detection is currently one of the most widely used tasks in the field of computer vision. This is primarily due to its broad range of practical applications, including security surveillance, vehicle detection, product quality inspection, traffic image analysis, and many others. Various architectures have been developed for object detection, ranging from classical Convolutional Neural Network (CNN)-based models and models from the YOLO family to more recent approaches based on Transformers and Vision Transformers. Each architecture has its own advantages and limitations in terms of accuracy, processing speed, training complexity, and practical deployment capability. 

- In this project, students will study and develop a practical application for object detection. Specifically, students will select a particular topic, construct a corresponding dataset, train and compare multiple model architectures on the same task, and then develop an application using the best-performing model. 

## **2 Requirements** 

1. **(8 points)** Students must select a specific topic for the object detection task and construct a corresponding dataset. For the selected task, students must train and evaluate at least three different model architectures for comparison. The report must clearly present the advantages and disadvantages of each model and analyze their accuracy, processing speed, and practical applicability. 

2. **(2 points)** Based on the experimental results obtained in Requirement 1, students must select the best-performing model and develop a web-based application that allows users to upload an image and returns the object detection results for that image. 

## **3 Detailed Requirements** 

- For Requirement 1, students may select any topic of interest, provided that the following requirements are satisfied: 

1 

- Students must submit the dataset used for model training. The dataset must contain at least five object classes. The number of data samples may vary depending on the selected task; however, students must collect sufficient data to ensure that model training and evaluation are meaningful. 

- Students must train and evaluate at least three different model architectures on the same task for comparison. Students are encouraged to select representative architectures from different approaches, such as: 

   - ∗ Traditional CNN-based approaches: Faster R-CNN, SSD, RetinaNet, etc. 

   - ∗ YOLO-based approaches: models from the YOLO family. 

   - ∗ Recent Transformer- or Vision Transformer-based approaches: DETR, Deformable DETR, ViTDet, etc. 

- Students must evaluate the models using appropriate evaluation metrics. The report must describe how the dataset is divided into training, validation, and test sets and compare the results obtained from the evaluated models. 

- The report must provide clear observations regarding the advantages and disadvantages of each architecture, including aspects such as accuracy, processing speed, training complexity, and practical applicability. 

## **4 Submission Guidelines** 

- Students must submit the complete source code, trained model weights, training dataset, and project report. 

- All submission materials must be packaged into a single ZIP file. The filename must contain the student IDs of all team members, separated by underscores, using the following format: 

```
StudentID1StudentID2...StudentIDN.zip
```

- If the submission file is too large, students may upload it to Google Drive and submit a text file containing the corresponding link. The text file must be named using the following format: 

```
StudentID1StudentID2...StudentIDN.txt
```

2 

