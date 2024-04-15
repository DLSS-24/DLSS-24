# Project Title: Analyzing Environmental Impact on Urban Landscapes with CNNs

## Overview
In this project, students will use Convolutional Neural Networks (CNNs) to analyze satellite images from different urban areas to assess environmental changes over time, such as green space reduction, urban heat islands, or water body changes. This project provides a hands-on opportunity to explore environmental issues within urban planning using deep learning.

### Tools and Technologies
- Python
- PyTorch for implementing the CNN model
- Pandas and NumPy for data manipulation
- Matplotlib and Seaborn for data visualization
- Satellite images available from open datasets

### Dataset
Students will use the "EuroSAT" dataset available on [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/eurosat) or similar open datasets that consist of satellite images labeled according to different land use types like forests, rivers, cities, and so on. This dataset is openly accessible and can be easily integrated into a deep learning project.

## Project Steps

### 1. Data Presentation and Preliminary Analysis (10 points)
- **Data Collection**: Download the EuroSAT dataset.
- **Data Cleaning**: Verify data integrity, handle any anomalies or missing data points.
- **Exploratory Data Analysis (EDA)**: Explore the dataset by visualizing different types of land cover, understand the distribution of classes, and analyze any temporal data if available.
- **Feature Engineering**: Discuss if additional features like temporal changes or augmented data could be useful.

### 2. Deep Learning Model Implementation (10 points)
- **Preprocessing**: Standardize the image sizes, normalize the pixel values, and organize data into training, validation, and test sets.
- **Model Architecture**: Develop a CNN in PyTorch with layers suitable for classifying land cover types.
- **Training the Model**: Train the model using the training data. Utilize the validation set for tuning hyperparameters and preventing overfitting.
- **Evaluation**: Use classification accuracy and other relevant metrics to evaluate model performance.

### 3. Research Analysis (10 points)
- **Change Detection Analysis**: Apply the trained model to detect changes in urban environments over available periods.
- **Impact Analysis**: Analyze how urban expansion has impacted green spaces, water bodies, or contributed to the urban heat island effect.
- **Visualization**: Create visual comparisons of urban development over time, highlighting environmental impacts.
- **Policy Recommendations**: Based on findings, suggest urban planning strategies to mitigate negative environmental impacts.

### 4. Presentation and Report Writing
- **Final Presentation**: Develop a presentation that outlines the project's scope, methodology, findings, and implications.
- **Report**: Write a comprehensive report that details every step of the project, including code, methodologies, findings, and future research directions.

## Expected Outcomes
This project aims to enhance students' understanding of CNN applications in real-world scenarios, focusing on environmental studies and urban planning. Students will gain practical skills in handling image data, training deep learning models, and interpreting the results to make informed conclusions about urban environmental changes.

### Example CNN Model in PyTorch
Here’s how the CNN model could be structured in PyTorch:

```python
import torch
import torch.nn as nn

class UrbanEnvCNN(nn.Module):
    def __init__(self):
        super(UrbanEnvCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(64 * 16 * 16, 10)  # Assuming 10 different classes of land cover

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten the output for Dense layer
        out = self.fc(out)
        return out

