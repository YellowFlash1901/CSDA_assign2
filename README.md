Project Name Road type classification.

Summary
This project explores using deep learning techniques to classify different road surface types (asphalt, gravel, concrete) from images. The project compares the performance of pre-trained models (VGG16, EfficientNetB0, DenseNet121) with customized versions that integrate spatial attention mechanisms. The goal is to achieve high classification accuracy while reducing the number of model parameters.

Team Members

Shayna Tuscano (23104937)
Vinit Vaidya (23101786)
Vardaan Sathe (23104093)
Varun Lokhande (23102661)
Siddharth Singh (23102447)
Abstract
This report describes a novel approach to classify road surface types using deep learning. A machine learning algorithm is implemented to distinguish between different road surface types based on image data. The report compares the performance of pre-trained models fine-tuned on the dataset with proposed customized model architectures that leverage attention mechanisms. The customized models aim to achieve better performance with a significant reduction in the number of parameters.

Introduction
The proposed model architecture utilizes pre-trained models like VGG16, EfficientNetB0, and DenseNet121, which are trained on large-scale image datasets like ImageNet. These pre-trained models are then customized by adding attention layers to focus on specific features in the images relevant to road surface classification. This approach aims to reduce the model size and improve performance compared to using the pre-trained models alone.

Literature Review
The report includes a review of relevant literature on:

Utilizing attention mechanisms in convolutional neural networks (CNNs) for image classification tasks. ([3])
Applying attention mechanisms in medical image analysis for segmentation tasks. ([4])
Leveraging deep learning for road type and quality recognition in Advanced Driver-Assistance Systems (ADAS). ([5])
Image-based road type classification using unsupervised feature learning and domain-engineered features. ([6])
Methodology
The report details three customized CNN models designed for road surface type classification. These models aim to learn features from labeled road surface images while reducing the number of parameters compared to pre-trained models.

The methodology involves two main stages:

Initial training with a custom classification head: This involves using pre-trained models as a foundation and adding a custom classification head with a dense layer and softmax activation for classification.
Integration of spatial attention mechanisms: To improve performance, spatial attention mechanisms are integrated into the models, focusing on relevant parts of the input data.
The report details the architecture integration process, including how the spatial attention function is incorporated into the models and how the overall architecture is compiled for training.

Dataset Used
The report describes the dataset used for training and testing the models. The dataset consists of images categorized into three classes: asphalt, gravel, and concrete. The report also details techniques used to address class imbalance and data pre-processing steps like downsizing images and normalization.

Experiments, Results & Analysis
The report presents the results obtained from training and evaluating the different models. It includes performance metrics like accuracy, precision, recall, and F1-score for both baseline pre-trained models and the improved versions with attention mechanisms. The results demonstrate that the customized models with attention mechanisms generally outperform the baseline models in terms of accuracy while having fewer parameters.

Conclusion
The report concludes by highlighting the limitations of the models, such as potential struggles with generalizing to unseen variations in road surface data and similar-looking classes. Future work directions are also suggested, including exploring different attention mechanisms, hyperparameter tuning, and applying the approach to other datasets and domains.

References
[1] https://www.hackersrealm.net/post/extract-features-from-image-python
[2] https://medium.com/@clairenyz/attention-based-convolutional-neural-network-a719693058a7
[3] Jetley, Saumya, et al. "Learn to pay attention." arXiv preprint arXiv:1804.02391 (2018).
[4] Oktay, Ozan, et al. "Attention u-net: Learning where to look for the pancreas. arXiv 2018." arXiv preprint arXiv:1804.03999 (1804).
[5] Tumen, V., Yildirim, O., & Ergen, B. (2018). Recognition of Road Type and Quality for Advanced Driver Assistance Systems with Deep Learning. Elektronika Ir Elektrotechnika, 24(6), 67-74
