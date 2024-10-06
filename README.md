# **Seismic Detection Across the Solar System**

## **Project Overview**

This project analyses the data from seismology missions and separates data about earthquakes from noise.

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [More info](#presentation)
5. [License](#license)

---

## **Overview**

1. **Noise Filtering:** First, the noise is removed by "Signal-to-Noise ratio" algorithm.  
This is done in [Data_Filter_Algorithm.py](Data_Filter_Algorithm.py) file.
2. **AI:** Then, a Bidirectional LSTM model processes the updated data and spots the earthquakes.  
The AI model is in [this directory](gangsta_car).
3. **Result**: In the end, the time, when the earthquake begins, is calculated from the output of AI.  
This is done in the [main file](main.py).
## **Installation**

### **Prerequisites**

- Python 3.9
- Required Python libraries:
  - numpy
  - pandas
  - scipy
  - obspy
  - matplotlib
  - scikit-learn
  - tqdm
  - torch
  - torchvision

### **Steps to Install**

1. Clone the repository:
    ```bash
    git clone https://github.com/Anton15K/Nasa-SpaceApp-Challenge-2024.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Nasa-SpaceApp-Challenge-2024
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## **Usage**

### **Step 1: Prepare Seismic Data**

Ensure your seismic data is in a CSV file with the following format:

| time_abs(%Y-%m-%dT%H:%M:%S.%f) | time_rel(sec) | velocity(m/s)  |
|--------------------------------|---------------|----------------|
| YYYY-MM-DD HH:MM:SS            | 0.0           | -6.1532789e-14 |
| YYYY-MM-DD HH:MM:SS            | 0.15          | -7.7012884e-14 |
| ...                            | ...           | ...            |

### **Step 2: change the file path**
In main.py, on 81 line, change the data_path variable to the path of your file and filename to the name of your file

### **Step 3: run main.py** 
To process the seismic data and filter out the noise:

```bash
python main.py
```

## **Presentation**
For more info, please refer to the presentation about this project.  
The presentation can be accessed by this link:
https://drive.google.com/file/d/1oU8SUEi75MI2FX7OAEqwaS13731-vfjo/view?usp=sharing

## **License**
This project is licensed under the Apache License. More information here: http://www.apache.org/licenses/
