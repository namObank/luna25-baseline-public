#!/usr/bin/env python3
"""
Analyze dataset: count unique subjects and find subject overlap
"""

import csv
from pathlib import Path
from collections import defaultdict

def analyze_dataset():
    # Setup paths
    script_dir = Path(__file__).resolve().parent.parent
    csv_dir = script_dir / "luna25-pulse-3d" / "dataset" / "luna25_csv"
    
    train_csv = csv_dir / "train.csv"
    valid_csv = csv_dir / "valid.csv"
    
    if not train_csv.exists() or not valid_csv.exists():
        print(f"❌ CSV files not found!")
        print(f"   Train: {train_csv}")
        print(f"   Valid: {valid_csv}")
        return
    
    # Load train data
    train_data = []
    with open(train_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotation_id = row['AnnotationID']
            label = int(row['label'])
            subject_id = annotation_id.split('_')[0]
            train_data.append({
                'annotation_id': annotation_id,
                'subject_id': subject_id,
                'label': label
            })
    
    # Load valid data
    valid_data = []
    with open(valid_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotation_id = row['AnnotationID']
            label = int(row['label'])
            subject_id = annotation_id.split('_')[0]
            valid_data.append({
                'annotation_id': annotation_id,
                'subject_id': subject_id,
                'label': label
            })
    
    # Extract unique subjects
    train_subjects = set(item['subject_id'] for item in train_data)
    valid_subjects = set(item['subject_id'] for item in valid_data)
    all_subjects = train_subjects | valid_subjects
    
    # Find overlap
    overlap = train_subjects & valid_subjects
    
    # Count samples
    train_malignant = sum(1 for item in train_data if item['label'] == 1)
    train_benign = len(train_data) - train_malignant
    
    valid_malignant = sum(1 for item in valid_data if item['label'] == 1)
    valid_benign = len(valid_data) - valid_malignant
    
    # Samples per subject
    train_per_subject = defaultdict(int)
    for item in train_data:
        train_per_subject[item['subject_id']] += 1
    
    valid_per_subject = defaultdict(int)
    for item in valid_data:
        valid_per_subject[item['subject_id']] += 1
    
    print("\n" + "="*80)
    print("📊 DATASET ANALYSIS - LUNA25")
    print("="*80)
    
    print(f"\n👥 SUBJECTS (Số người tham gia)")
    print(f"  Total unique subjects:    {len(all_subjects)} người")
    print(f"  Subjects in TRAIN:        {len(train_subjects)} người")
    print(f"  Subjects in VALID:        {len(valid_subjects)} người")
    print(f"  Subjects in BOTH:         {len(overlap)} người ⚠️  (DATA LEAK!)")
    
    print(f"\n📦 SAMPLES (Số mẫu quét)")
    print(f"  Train samples:            {len(train_data)} samples")
    print(f"    - Benign (0):           {train_benign} ({100*train_benign/len(train_data):.1f}%)")
    print(f"    - Malignant (1):        {train_malignant} ({100*train_malignant/len(train_data):.1f}%)")
    
    print(f"\n  Valid samples:            {len(valid_data)} samples")
    print(f"    - Benign (0):           {valid_benign} ({100*valid_benign/len(valid_data):.1f}%)")
    print(f"    - Malignant (1):        {valid_malignant} ({100*valid_malignant/len(valid_data):.1f}%)")
    
    print(f"\n  Total:                    {len(train_data) + len(valid_data)} samples")
    
    print(f"\n⚠️  OVERLAP ANALYSIS (Samples trùng người)")
    if len(overlap) > 0:
        overlap_samples = 0
        overlap_details = {}
        
        for subject_id in overlap:
            train_count = train_per_subject[subject_id]
            valid_count = valid_per_subject[subject_id]
            overlap_samples += train_count + valid_count
            overlap_details[subject_id] = (train_count, valid_count)
        
        print(f"  ❌ DATA LEAK DETECTED!")
        print(f"  Subjects appear in BOTH train and valid: {len(overlap)}")
        print(f"  Total samples affected: {overlap_samples}")
        
        print(f"\n  Top 10 subjects with samples in both sets:")
        sorted_overlap = sorted(
            overlap_details.items(),
            key=lambda x: x[1][0] + x[1][1],
            reverse=True
        )
        for i, (subject_id, (train_count, valid_count)) in enumerate(sorted_overlap[:10], 1):
            total = train_count + valid_count
            print(f"    {i}. Subject {subject_id}: {train_count} in train + {valid_count} in valid = {total} samples")
    else:
        print(f"  ✅ No overlap - CLEAN split!")
    
    print(f"\n📈 SAMPLES PER SUBJECT")
    max_train = max(train_per_subject.values())
    min_train = min(train_per_subject.values())
    avg_train = sum(train_per_subject.values()) / len(train_per_subject)
    
    max_valid = max(valid_per_subject.values())
    min_valid = min(valid_per_subject.values())
    avg_valid = sum(valid_per_subject.values()) / len(valid_per_subject)
    
    print(f"  TRAIN:")
    print(f"    - Min samples per subject:  {min_train}")
    print(f"    - Max samples per subject:  {max_train}")
    print(f"    - Avg samples per subject:  {avg_train:.1f}")
    
    print(f"  VALID:")
    print(f"    - Min samples per subject:  {min_valid}")
    print(f"    - Max samples per subject:  {max_valid}")
    print(f"    - Avg samples per subject:  {avg_valid:.1f}")
    
    print(f"\n🏆 TOP SUBJECTS BY NUMBER OF SAMPLES")
    all_subject_counts = defaultdict(int)
    for item in train_data + valid_data:
        all_subject_counts[item['subject_id']] += 1
    
    sorted_subjects = sorted(all_subject_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (subject_id, count) in enumerate(sorted_subjects[:10], 1):
        in_train = train_per_subject.get(subject_id, 0)
        in_valid = valid_per_subject.get(subject_id, 0)
        marker = "⚠️ " if (in_train > 0 and in_valid > 0) else "✓ "
        print(f"    {i}. Subject {subject_id}: {count:3d} samples ({marker}{in_train} train, {in_valid} valid)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_dataset()
