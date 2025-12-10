#!/usr/bin/env python3
"""
Simple script to show a single sample data
"""

import numpy as np
from pathlib import Path
import sys
import csv

def main():
    # Setup paths
    script_dir = Path(__file__).resolve().parent.parent
    dataset_dir = script_dir / "dataset"
    
    # Check if dataset exists
    if not (dataset_dir / "image").exists():
        print("❌ Dataset not found!")
        sys.exit(1)
    
    print(f"✓ Dataset found at: {dataset_dir}")
    
    # Find first sample from train CSV
    csv_path = script_dir / "luna25-pulse-3d" / "dataset" / "luna25_csv" / "train.csv"
    
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        sys.exit(1)
    
    # Read first sample ID from train CSV
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
        sample_id = first_row['AnnotationID']
        label = int(first_row['label'])
    
    print(f"\n🔍 Loading first training sample: {sample_id}")
    print(f"   Label: {label} (0=benign, 1=malignant)")
    
    # Load image
    image_file = dataset_dir / "image" / f"{sample_id}.npy"
    if not image_file.exists():
        print(f"❌ Image file not found: {image_file}")
        sys.exit(1)
    
    image = np.load(image_file)
    
    # Parse annotation ID
    parts = sample_id.split('_')
    subject_id = parts[0]
    index = parts[1]
    date = parts[2] if len(parts) > 2 else "N/A"
    
    print("\n" + "="*70)
    print("📊 SAMPLE DETAILS")
    print("="*70)
    
    print(f"\n🔍 ANNOTATION INFO")
    print(f"  Subject ID:        {subject_id} (bệnh nhân)")
    print(f"  Annotation Index:  {index} (lần ghi nhận thứ {index})")
    print(f"  Scan Date:         {date}")
    print(f"  Label:             {label} ({'lành' if label == 0 else 'ác/ung thư'})")
    
    print(f"\n📈 IMAGE (CT SCAN 3D)")
    print(f"  Shape:             {image.shape}")
    print(f"    - Depth (z):     {image.shape[0]} slices (mỏng)")
    print(f"    - Height (y):    {image.shape[1]} pixels (chiều cao)")
    print(f"    - Width (x):     {image.shape[2]} pixels (chiều rộng)")
    print(f"  Data type:         {image.dtype}")
    print(f"  Total voxels:      {np.prod(image.shape):,} voxels")
    
    print(f"\n📊 VALUE STATISTICS (Hounsfield Units)")
    print(f"  Min value:         {image.min():.0f} HU (không khí/ngoài cơ thể)")
    print(f"  Max value:         {image.max():.0f} HU (xương/mô đặc)")
    print(f"  Mean:              {image.mean():.0f} HU (trung bình)")
    print(f"  Median:            {np.median(image):.0f} HU (trung vị)")
    print(f"  Std Dev:           {image.std():.0f} HU (độ lệch chuẩn)")
    
    # Find nonzero region
    nonzero_count = np.count_nonzero(image != image.min())
    total = np.prod(image.shape)
    print(f"  Non-background:    {nonzero_count:,} voxels ({100*nonzero_count/total:.1f}%)")
    
    print(f"\n🫁 GIẢI THÍCH DỮ LIỆU")
    print(f"  - Đây là một CROP từ CT scan phổi bệnh nhân {subject_id}")
    print(f"  - Kích thước: 64×128×128 voxel (~500KB)")
    print(f"  - Mục đích: Phân loại nốt này là lành hay ác")
    print(f"  - Model sẽ xem 3D volume này và dự đoán nhãn")
    
    print(f"\n📍 HOUNSFIELD UNIT REFERENCE")
    print(f"  HU < -500  → Không khí (ngoài cơ thể)")
    print(f"  -1000 ~ -500 → Phổi bình thường")
    print(f"  -100 ~ 0   → Mô mềm")
    print(f"  0 ~ 100    → Máu, mô dày đặc")
    print(f"  > 400      → Xương")
    
    # Show some slices
    print(f"\n🖼️  SAMPLE SLICES (center slices)")
    d, h, w = image.shape
    cd, ch, cw = d//2, h//2, w//2
    
    axial = image[cd, :, :]
    coronal = image[:, ch, :]
    sagittal = image[:, :, cw]
    
    print(f"\n  AXIAL SLICE (z={cd}) - nhìn từ trên xuống")
    print(f"    Shape: {axial.shape}, Range: [{axial.min():.0f}, {axial.max():.0f}]")
    
    print(f"\n  CORONAL SLICE (y={ch}) - nhìn từ trước ra sau")
    print(f"    Shape: {coronal.shape}, Range: [{coronal.min():.0f}, {coronal.max():.0f}]")
    
    print(f"\n  SAGITTAL SLICE (x={cw}) - nhìn từ phải sang trái")
    print(f"    Shape: {sagittal.shape}, Range: [{sagittal.min():.0f}, {sagittal.max():.0f}]")
    
    print(f"\n" + "="*70)
    print(f"✓ Sample analysis complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
