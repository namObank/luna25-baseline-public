📦 LUNA25 Baseline Algorithm

1. Prepare environment and test the code

Install python 3.10

Note: first, copy the resource folder containing *pth files into this repo

Install packages

```bash
pip install -r requirements.txt
```

To test the code, run server directly from source
```bash
uvicorn server:app --reload
```
Alternatively
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Import cURL in postman to send request (remember to include mha file in the request)
```bash
curl --location 'localhost:8000/api/v1/predict/lesion' \
--form 'file=@"/D:/luna_data/luna25_images_sample/1.3.6.1.4.1.14519.5.2.1.7009.9004.302654768337221344067573753621.mha"' \
--form 'seriesInstanceUID="1.3.6.1.4.1.14519.5.2.1.7009.9004.302654768337221344067573753621"' \
--form 'patientID="212849"' \
--form 'studyDate="20000102"' \
--form 'lesionID="2"' \
--form 'coordX="108.05"' \
--form 'coordY="67.82"' \
--form 'coordZ="-227.01"' \
--form 'ageAtStudyDate="72"' \
--form 'gender="Female"'
```

2. Containerizing 

Packaging
```bash
docker build -t <image_name> .
```

Running (specify your port as needed)
```bash
docker run -it --rm -p 8000:8000 <image_name>
```

Dưới đây là một mẫu **README** mà ông có thể dùng cho project FastAPI, hướng dẫn từ cài đặt môi trường Conda, Linux, Python 3.10, đến build và chạy Docker cho mục đích inference:

---

# Project Name

Mục tiêu: Triển khai FastAPI để phục vụ **inference** mô hình.

## 1. Yêu cầu hệ thống

* Hệ điều hành: Linux (Ubuntu 20.04+ khuyến nghị)
* Python: 3.10
* Conda: >= 4.10
* Docker & Docker Compose: >= 20

---

## 2. Cài đặt môi trường Conda

1. Tạo môi trường Conda với Python 3.10:

```bash
conda create -n myenv python=3.10 -y
```

2. Kích hoạt môi trường:

```bash
conda activate myenv
```

3. Cài đặt các dependencies:

```bash
pip install -r requirements.txt
```

> Ghi chú: `requirements.txt` nên chứa các thư viện như `fastapi`, `uvicorn`, và các thư viện inference của mô hình bạn.

---

## 3. Chạy server FastAPI

Trong môi trường Conda đã kích hoạt:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

* `app.main:app` là đường dẫn tới `FastAPI()` instance trong project của ông.
* Mở trình duyệt hoặc postman truy cập: `http://localhost:8000/docs` để xem API docs.

---

## 4. Chạy inference trực tiếp

Ví dụ gọi API `POST /predict`:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"input": "your input data"}'
```

---

## 5. Build Docker Image

1. Tạo file `Dockerfile` ví dụ:

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port
EXPOSE 8000

# Lệnh chạy server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. Build Docker image:

```bash
docker build -t fastapi-inference:latest .
```

3. Chạy Docker container:

```bash
docker run -d -p 8000:8000 fastapi-inference:latest
```

* Truy cập: `http://localhost:8000/docs`

---

## 6. Optional: Docker Compose

Tạo `docker-compose.yml`:

```yaml
version: "3.9"

services:
  fastapi-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
```

Chạy:

```bash
docker-compose up --build
```

---

## 7. Kết luận

* Môi trường Conda giúp quản lý Python 3.10 và dependencies.
* Docker giúp deploy nhanh và nhất quán trên Linux server.
* Server FastAPI sẵn sàng phục vụ **inference** thông qua API endpoint.

---

Nếu muốn, tôi có thể viết luôn **phiên bản README tối giản, chuẩn công ty**, vừa dễ copy, vừa đủ chạy inference trên Docker mà không cần nhiều giải thích. Ông có muốn tôi làm luôn không?


<!-- # 📦 LUNA25 Baseline Algorithm
Thank you for participating in the [LUNA25 Challenge](https://luna25.grand-challenge.org/).

In LUNA25, we want to use artificial intelligence for lung nodule malignancy risk estimation on low-dose chest CT scans. For this, we have prepared two baseline models (2D and 3D model) that can help you get started. 

The development of your algorithms should be performed using your local GPU or a cloud platform (such as AWS or Azure), while algorithm evaluation will be performed exclusively on the [Grand-Challenge](https://grand-challenge.org/) platform.

## 🗂️ Content
This baseline algorithm provides a framework for training and testing models. While it includes basic scripts, we encourage you to extend and customize them to develop alternative or improved methods.

Important Files:
- 🦾 `train.py`: A script for training the baseline algorithm on local data.
- 🦿 `inference.py`: A script for testing the trained algorithm using a specified configuration.
- 🧮 `Dockerfile`: A file to build a Docker container for deployment on Grand-Challenge. For help on setting up Docker with GPU support you can check the documentation on [Grand-Challenge](https://grand-challenge.org/documentation/setting-up-wsl-with-gpu-support-for-windows-11/) or [Docker](https://docs.docker.com/engine/install/ubuntu/) for additional information.

## ⚙️ Setting up the Environment
To set up the required environment for the baseline algorithm:
1. **Create an environment and esure Python is Installed**: Install Python 3.9 or higher:
    ```bash
    conda create -n luna25-baseline python==3.9
    ```
2. **Install Dependencies**:
    - Run the following command to install the dependencies listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
3. **Verify Installation**:
    - Test the installation by running:
    ```bash
    python --version
    pip list
    ```
    Ensure all required packages are listed and no errors are reported.

## 🚀 Performing a Training Run
1. **Set up training configurations**

Open `experiment_config.py` to edit your training configurations. Key parameters include:

- `self.MODE`: Set this to 2D or 3D depending on the desired baseline model.
- `self.EXPERIMENT_NAME`: Specify the name of your experiment (e.g. LUNA25-baseline).
- `self.CSV_DIR_TRAIN`: the path to the training csv file
- `self.DATADIR`: the path where the images are stored


2. **Training the Model**

To train the model using the `train.py` script:
```bash
python train.py
```
This script uses the settings from experiment_config.py to initialize and train the model.

## 🧪 Testing the Trained Algorithm
1. **Configure the inference script**

Open the `inference.py` script and configure:
- `INPUT_PATH`: Path to the input data (CT, nodule locations and clinical information). Keep as `Path("/input")` for Grand-Challenge.
- `RESOUCE_PATH`: Path to resources (e.g., pretrained models weights) in the container. Defaults to `/results` directory (see Dockerfile)
- `OUTPUT_PATH`: Path to store the output in your local directory. Keep as `Path("/output")` for Grand-Challenge.
- **Inputs for the `run()` function**:
    - `mode`: Match this to the mode used during training (2D or 3D).
    - `model_name`: Specify the experiment_name matching the training configuration (corresponding to experiment_name directory that contains the model weights in `/results`).

2. **Updating the Docker Image Tag**

In `do_test_run.sh`, update the Docker image tag as needed:
```bash
DOCKER_IMAGE_TAG="luna25-baseline-3d-algorithm-open-development-phase"
```


3. **Running the Test Script**

To test the trained model for running inference run: 
```bash
./do_test_run.sh
``` 

This script performs the following:
- Uses Docker to execute the `inference.py` script.
- Mounts necessary input and output directories.
- Adjusts the Docker image tag (if updated) before running.

## 🐳 Building the Docker Image
To build the Docker container required for submission to Grand-Challenge run:
```bash
./do_save.sh
```
This will output a *.tar.gz file, which can be uploaded to Grand-Challenge.
More information on testing and deploying your container can be found [here](https://grand-challenge.org/documentation/test-and-deploy-your-container/).

## 🛠️ Extending the Baseline
While this baseline provides a starting point, participants are encouraged to:

- Implement advanced AI models.
- Explore alternative data preprocessing and augmentation techniques.
- perform Ensemble Learning
- train models using entire or larger CT scan inputs

For questions, refer to the [LUNA25 Challenge Page](https://luna25.grand-challenge.org/).

Good luck! -->