#!/usr/bin/env python3
"""
Generate proper train/valid split using GroupKFold (by subject)
Ensures NO patient appears in both training and validation sets
"""

import csv
from pathlib import Path
from collections import defaultdict
import sys

def load_labels(labels_csv):
    """Load labels from original CSV"""
    labels = {}
    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotation_id = row['AnnotationID']
            label = int(row['label'])
            labels[annotation_id] = label
    return labels

def group_kfold_split(annotation_ids, labels_dict, n_splits=5, fold=0):
    """
    Implement GroupKFold split manually (without sklearn)
    Groups by subject ID (first part of annotation_id)
    
    Args:
        annotation_ids: list of annotation IDs
        labels_dict: dict of label values
        n_splits: number of folds
        fold: which fold to use as validation (0-4)
    
    Returns:
        (train_ids, valid_ids)
    """
    
    # Group by subject
    subject_to_samples = defaultdict(list)
    for ann_id in annotation_ids:
        subject_id = ann_id.split('_')[0]
        subject_to_samples[subject_id].append(ann_id)
    
    # Get unique subjects
    unique_subjects = sorted(subject_to_samples.keys())
    n_subjects = len(unique_subjects)
    
    print(f"\nℹ️  GroupKFold Parameters:")
    print(f"  Total subjects: {n_subjects}")
    print(f"  Total samples: {len(annotation_ids)}")
    print(f"  Number of folds: {n_splits}")
    print(f"  Using fold: {fold} for validation")
    
    # Calculate fold size
    fold_size = n_subjects // n_splits
    remainder = n_subjects % n_splits
    
    # Distribute subjects to folds
    fold_assignments = []
    subject_idx = 0
    for f in range(n_splits):
        # Larger folds get one extra subject to handle remainder
        size = fold_size + (1 if f < remainder else 0)
        fold_subjects = unique_subjects[subject_idx:subject_idx + size]
        fold_assignments.extend([(subj, f) for subj in fold_subjects])
        subject_idx += size
    
    # Split into train and valid
    train_ids = []
    valid_ids = []
    
    for subject_id, fold_id in fold_assignments:
        samples = subject_to_samples[subject_id]
        if fold_id == fold:
            valid_ids.extend(samples)
        else:
            train_ids.extend(samples)
    
    return train_ids, valid_ids

def save_csv(annotation_ids, labels_dict, output_path):
    """Save annotation IDs and labels to CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['AnnotationID', 'label'])
        writer.writeheader()
        for ann_id in annotation_ids:
            writer.writerow({
                'AnnotationID': ann_id,
                'label': labels_dict[ann_id]
            })

def main():
    # Setup paths
    script_dir = Path(__file__).resolve().parent.parent
    labels_file = script_dir / "TMHOA" / "Kaggle" / "LUNA25_Public_Training_Development_Data.csv"
    
    # Try alternate path
    if not labels_file.exists():
        labels_file = script_dir / "LUNA25_Public_Training_Development_Data.csv"
    
    if not labels_file.exists():
        # Search for it
        import glob
        matches = glob.glob(str(script_dir / "**" / "LUNA25_Public_Training_Development_Data.csv"), recursive=True)
        if matches:
            labels_file = Path(matches[0])
            print(f"Found labels file: {labels_file}")
        else:
            print(f"❌ Labels file not found!")
            print(f"Searched: {script_dir}")
            return
    
    print(f"✓ Loading labels from: {labels_file}")
    labels_dict = load_labels(labels_file)
    print(f"✓ Loaded {len(labels_dict)} annotations")
    
    # Get all annotation IDs
    all_annotation_ids = sorted(labels_dict.keys())
    
    # Create output directory
    output_dir = Path(__file__).resolve().parent.parent / "luna25-baseline-public" / "dataset" / "luna25_csv_groupkfold"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n" + "="*80)
    print("📊 GENERATING 5-FOLD CROSS-VALIDATION SPLITS")
    print("="*80)
    
    # Generate all 5 folds
    for fold in range(5):
        print(f"\n🔄 Fold {fold + 1}/5:")
        train_ids, valid_ids = group_kfold_split(
            all_annotation_ids,
            labels_dict,
            n_splits=5,
            fold=fold
        )
        
        # Save CSVs
        train_csv = output_dir / f"train_fold{fold}.csv"
        valid_csv = output_dir / f"valid_fold{fold}.csv"
        
        save_csv(train_ids, labels_dict, train_csv)
        save_csv(valid_ids, labels_dict, valid_csv)
        
        # Statistics
        train_mal = sum(1 for aid in train_ids if labels_dict[aid] == 1)
        valid_mal = sum(1 for aid in valid_ids if labels_dict[aid] == 1)
        
        train_subjects = set(aid.split('_')[0] for aid in train_ids)
        valid_subjects = set(aid.split('_')[0] for aid in valid_ids)
        overlap = train_subjects & valid_subjects
        
        print(f"  Train: {len(train_ids):5d} samples ({len(train_subjects):4d} subjects, {train_mal:3d} malignant)")
        print(f"  Valid: {len(valid_ids):5d} samples ({len(valid_subjects):4d} subjects, {valid_mal:3d} malignant)")
        print(f"  Overlap: {len(overlap)} subjects - {'✅ CLEAN' if len(overlap) == 0 else '❌ LEAK'}")
        print(f"  📁 Saved to:")
        print(f"     - {train_csv}")
        print(f"     - {valid_csv}")
    
    print(f"\n" + "="*80)
    print(f"✅ Generated 5-fold cross-validation splits!")
    print(f"📁 Location: {output_dir}")
    print(f"\nUsage:")
    print(f"  for fold in range(5):")
    print(f"      train_df = pd.read_csv(f'train_fold{{fold}}.csv')")
    print(f"      valid_df = pd.read_csv(f'valid_fold{{fold}}.csv')")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
