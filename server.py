from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import SimpleITK as sitk
import tempfile
import os
from preprocessing.extract_nodules import NoduleExtractor
from pathlib import Path
from preprocessing import utils
from preprocessing.convert2nodule_block import NodulePreProcessor
from experiment_config import Configuration
from dataloader import get_data_loader
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.model_2d import ResNet34
from models.i3d import DualPathI3DNet
from models.model_3d_resnet import LungNodule3DResNet
from models.model_3d_resnetSE import LungNodule3DSEResNet
from models.dual_path_model import DualPathLungNoduleNet
import time
from models.transformerI3D import TransformerI3D

app = FastAPI()


# device = torch.device("cuda:0")
device = "cpu"

# Khởi tạo Dual I3D
# model = DualPathI3DNet(
#     num_classes=1, 
#     input_channels=1 # Input thực tế là 1 kênh, I3D sẽ tự expand
# ).to(device)

# TRANSFORMER I3D dùng cái này
model_path = os.path.join("resources", "transformerI3D_F0.pth")
model = TransformerI3D(
    num_classes=1, 
    input_channels=1
).to(device)
if os.path.exists(model_path):
    # Dùng map_location vì device hiện tại là cpu
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Successfully loaded weights from: {model_path}")
else:
    print(f"WARNING: Weights file not found at {model_path}")
    
@app.post("/api/v1/predict/lesion")
async def upload_files(
    # csv_file: UploadFile = File(...),
    file: UploadFile = File(...),
    seriesInstanceUID = Form(...),
    patientID = Form(...),
    studyDate = Form(...),
    lesionID = Form(...),
    coordX = Form(...),
    coordY = Form(...),
    coordZ = Form(...),
    ageAtStudyDate = Form(...),
    gender = Form(...),
):

    # Parse command line arguments
    # parser = argparse.ArgumentParser(description="Train baseline model with 5-fold CV")
    # parser.add_argument("--fold", type=int, default=None, help="Specific fold to run (0-4). If None, runs all 5 folds.")
    # parser.add_argument("--mode", type=str, default="2D", choices=["2D", "3D"], help="Model mode: 2D or 3D")
    # args = parser.parse_args()

    # Kiểm tra file CSV
    # if not csv_file.filename.endswith(".csv"):
    #     return {"error": "File CSV không hợp lệ"}
    # df = pd.read_csv(csv_file.file)
    csv_path = './LUNA25_Public_Training_Development_Data.csv'
    df = pd.read_csv(csv_path)

    inference_item_df = df[df['SeriesInstanceUID'] == seriesInstanceUID]
    print(inference_item_df)

    # Kiểm tra file MHA
    if not file.filename.endswith(".mha"):
        return {"error": "File MHA không hợp lệ"}
    

    original_filename = file.filename  # ví dụ: "scan1.mha"
    _, ext = os.path.splitext(original_filename)

    # Tạo thư mục tạm (tempfile tự quản lý)
    # with tempfile.TemporaryDirectory() as tmp_dir:
    #     tmp_path = os.path.join(tmp_dir, original_filename)  # tên giống file gốc
    #     # Ghi nội dung upload vào tmp file
    #     with open(tmp_path, "wb") as f:
    #         f.write(file.file.read())

    mha_dir = 'raw_data/mha'
    os.makedirs(mha_dir, exist_ok=True)
    mha_path = os.path.join(mha_dir, original_filename)
    with open(mha_path, "wb") as f:
        f.write(file.file.read())

    # Đọc bằng SimpleITK
    image = sitk.ReadImage(mha_path)

    nii_path = 'nii_folder'

    # Xuất nodule ra nii
    extractor = NoduleExtractor(
        csv_path=Path(
            csv_path
        ),
        image_root=Path(mha_dir),
        output_path=Path(
            nii_path
        ),
        postfix="_0000",
        save_format=".nii.gz",
    )

    extractor.process_seriesuid(seriesInstanceUID)
    # extractor.dataset.SeriesInstanceUID.unique(),

    npy_path = "dataset/luna25_nodule_blocks"
    
    # Create config with the specified mode
    config = Configuration(mode='3D', data_dir=npy_path)

    # Chuyển đổi nii sang npy
    preprocessor = NodulePreProcessor(
        data_path=Path(nii_path),
        csv_path=Path(csv_path),
        save_path=Path(npy_path),
    )

    # preprocessor.dataset.AnnotationID.unique()

    annotation_ids = [f"{patientID}_{lesionID}_{studyDate}" for lesionID in range(1, 4)]
    for annotation_id in annotation_ids:
        preprocessor.prepare_numpy_files(annotation_id)

    # """
    # Khởi tạo dataloader
    loader = get_data_loader(
        config.DATADIR,
        inference_item_df,
        mode=config.MODE,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=None,
        translations=None,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    start = time.perf_counter()


    # inference với model
    # model = ResNet34().to(device)

    
    index = 0
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.float32, device=device)
        for val_data in loader:
            index += 1
            print(f"Infering item {index}")
            # val_images, val_labels = (
            #     val_data["image"].to(device),
            #     val_data["label"].to(device),
            # )
            # val_images = val_images.to(device)
            # val_labels = val_labels.float().to(device)
            # outputs = model(val_images)
            # # loss = loss_function(outputs.squeeze(), val_labels.squeeze())
            # loss = loss_function(outputs, val_labels.float())
            
            # 
            val_local = val_data["image_local"].to(device)
            val_global = val_data["image_global"].to(device)
            val_labels = val_data["label"].float().to(device)
            # outputs = model(val_local, val_global)
            # TRANSFORMER I3D dùng cái này
            outputs = model(val_local)

            # loss = loss_function(outputs, val_labels)
            
            # epoch_loss += loss.item()
            y_pred = torch.cat([y_pred, outputs], dim=0)
            y = torch.cat([y, val_labels], dim=0)
            print(f"Result {y}")
    # """

    # Trả ra kết quả
    lesionIdInt = int(lesionID)
    if lesionIdInt >= y_pred.size(0):
        raise TypeError("Index cho tensor 1D phải là int")

    pred_label = int(y[lesionIdInt].item())
    prob = y_pred[lesionIdInt].item()


    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000  # to milliseconds
    print(f"Execution time: {elapsed_ms:.3f} ms")

    return {
        "status": "success",
        "data": {
            "seriesInstanceUID": seriesInstanceUID,
            "lesionID": lesionIdInt,
            "probability": prob,
            "predictionLabel": pred_label,
            "processingTimeMs": int(elapsed_ms)
        }
    }


