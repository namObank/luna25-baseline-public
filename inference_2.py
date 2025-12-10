""""
Script for training a ResNet34 or I3D to classify a pulmonary nodule as benign or malignant.
"""
from models.model_2d import ResNet34
from models.i3d import DualPathI3DNet
from models.model_3d_resnet import LungNodule3DResNet
from models.model_3d_resnetSE import LungNodule3DSEResNet
from models.dual_path_model import DualPathLungNoduleNet
from dataloader import get_data_loader
from experiment_config import Configuration
import logging
import numpy as np
import torch
import sklearn.metrics as metrics
from tqdm import tqdm
import warnings
import random
import pandas
from datetime import datetime
import argparse
from pathlib import Path
import json
import csv
from monai.losses import FocalLoss 

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

def make_weights_for_balanced_classes(labels):
    """Making sampling weights for the data samples
    :returns: sampling weights for dealing with class imbalance problem
    """
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))

    weights = []
    for label in labels:
        weights.append(n_samples / float(cnt_dict[label]))
    return weights


def train(
    train_csv_path,
    valid_csv_path,
    exp_save_root,
    config,
):
    """
    Train a ResNet34 or an I3D model
    Returns: dict with training metrics
    """
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    logging.info(f"Training with {train_csv_path}")
    logging.info(f"Validating with {valid_csv_path}")

    train_df = pandas.read_csv(train_csv_path)
    valid_df = pandas.read_csv(valid_csv_path)
    
    # Initialize metrics tracker
    train_metrics = {
        "train_csv": str(train_csv_path),
        "valid_csv": str(valid_csv_path),
        "best_auc": -1,
        "best_final_score": -1,
        "best_sensitivity": -1,
        "best_specificity": -1,
        "best_epoch": -1,
        "total_epochs": 0,
        "train_samples": len(train_df),
        "valid_samples": len(valid_df),
        "malignant_train": int(train_df.label.sum()),
        "benign_train": int(len(train_df) - train_df.label.sum()),
        "malignant_valid": int(valid_df.label.sum()),
        "benign_valid": int(len(valid_df) - valid_df.label.sum()),
    }

    logging.info(
        f"Number of malignant training samples: {train_df.label.sum()}"
    )
    logging.info(
        f"Number of benign training samples: {len(train_df) - train_df.label.sum()}"
    )
    print()
    logging.info(
        f"Number of malignant validation samples: {valid_df.label.sum()}"
    )
    logging.info(
        f"Number of benign validation samples: {len(valid_df) - valid_df.label.sum()}"
    )

    # create a training data loader
    weights = make_weights_for_balanced_classes(train_df.label.values)
    weights = torch.DoubleTensor(weights)
    
    # --- Configurable Sampler ---
    # Uncomment the line below to use WeightedRandomSampler (Balanced training)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_df))
    
    # Uncomment the line below to use Standard training (No sampler, standard shuffle)
    # sampler = None 
    
    train_loader = get_data_loader(
        config.DATADIR,
        train_df,
        mode=config.MODE,
        sampler=sampler,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    valid_loader = get_data_loader(
        config.DATADIR,
        valid_df,
        mode=config.MODE,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=None,
        translations=None,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    device = torch.device("cuda:0")

    if config.MODE == "2D":
        model = ResNet34().to(device)
    # elif config.MODE == "3D":
    #     model = LungNodule3DSEResNet(num_classes=1, input_channels=1, pretrained_path=config.MODEL_3D_RESNET).to(device)
    # elif config.MODE == "3D":
    #     model = LungNodule3DResNet(
    #         pretrained_path=config.MODEL_3D_RESNET # Đường dẫn đến file .pth MedicalNet
    #     ).to(device)

    # if config.MODE == "3D":
    #     model = DualPathLungNoduleNet(
    #         num_classes=1,
    #         pretrained_path=config.MODEL_3D_RESNET
    #     ).to(device)
    
    if config.MODE == "3D":
        # Khởi tạo Dual I3D
        model = DualPathI3DNet(
            num_classes=1, 
            input_channels=1 # Input thực tế là 1 kênh, I3D sẽ tự expand
        ).to(device)

    # loss_function = torch.nn.BCEWithLogitsLoss()

    loss_function = FocalLoss(
        gamma=2.0,
        reduction='mean',
        use_softmax=False, 
        to_onehot_y=False
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # start a typical PyTorch training
    best_metric = -1 # This tracks best AUC
    best_snapshot = {} # Stores other metrics at the time of best AUC
    best_metric_epoch = -1
    epochs = config.EPOCHS
    patience = config.PATIENCE
    counter = 0

    for epoch in range(epochs):

        if counter > patience:
            logging.info(f"Model not improving for {patience} epochs")
            break

        logging.info("-" * 10)
        logging.info("epoch {}/{}".format(epoch + 1, epochs))

        # train

        # model.train()

        # epoch_loss = 0
        # step = 0

        """
        for batch_data in tqdm(train_loader):
            step += 1
            # inputs, labels = batch_data["image"], batch_data["label"]
            # labels = labels.float().to(device)
            # inputs = inputs.to(device)
            # optimizer.zero_grad()
            # outputs = model(inputs)

            input_local = batch_data["image_local"].to(device)
            input_global = batch_data["image_global"].to(device)
            labels = batch_data["label"].float().to(device)
            optimizer.zero_grad()
            outputs = model(input_local, input_global)

            # loss = loss_function(outputs.squeeze(), labels.squeeze())
            loss = loss_function(outputs, labels.float()) 
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_len = len(train_df) // train_loader.batch_size
            if step % 100 == 0:
                logging.info(
                    "{}/{}, train_loss: {:.4f}".format(step, epoch_len, loss.item())
                )
        epoch_loss /= step
        logging.info(
            "epoch {} average train loss: {:.4f}".format(epoch + 1, epoch_loss)
        )
        """

        # validate
        

        model.eval()

        epoch_loss = 0
        step = 0

        with torch.no_grad():

            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.float32, device=device)
            for val_data in valid_loader:
                step += 1
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
                outputs = model(val_local, val_global)
                loss = loss_function(outputs, val_labels)
                
                epoch_loss += loss.item()
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)

                epoch_len = len(valid_df) // valid_loader.batch_size

            epoch_loss /= step
            logging.info(
                "epoch {} average valid loss: {:.4f}".format(epoch + 1, epoch_loss)
            )

            # --- Calculate Metrics ---
            y_pred_prob = torch.sigmoid(y_pred.reshape(-1)).data.cpu().numpy().reshape(-1)
            y_true = y.data.cpu().numpy().reshape(-1)

            # 1. AUC
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_prob)
            auc_metric = metrics.auc(fpr, tpr)

            # 2. Sensitivity & Specificity (Threshold 0.5)
            y_pred_binary = (y_pred_prob > 0.5).astype(int)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_binary).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            final_score = auc_metric # Assuming Final Score is AUC

            if auc_metric > best_metric:

                counter = 0
                best_metric = auc_metric
                best_metric_epoch = epoch + 1
                
                # Snapshot other metrics at this best moment
                best_snapshot = {
                    "final_score": final_score,
                    "sensitivity": sensitivity,
                    "specificity": specificity
                }

                torch.save(
                    model.state_dict(),
                    exp_save_root / "best_metric_model.pth",
                )

                metadata = {
                    "train_csv": train_csv_path,
                    "valid_csv": valid_csv_path,
                    "config": config,
                    "best_auc": best_metric,
                    "final_score": final_score,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "epoch": best_metric_epoch,
                }
                np.save(
                    exp_save_root / "config.npy",
                    metadata,
                )

                logging.info("saved new best metric model")

            logging.info(
                "epoch: {} | AUC: {:.4f} (Best: {:.4f}) | Sens: {:.4f} | Spec: {:.4f}".format(
                    epoch + 1, auc_metric, best_metric, sensitivity, specificity
                )
            )
        counter += 1

    logging.info(
        "train completed, best_auc: {:.4f} at epoch: {}".format(
            best_metric, best_metric_epoch
        )
    )
    
    # Update metrics and return
    train_metrics["best_auc"] = float(best_metric)
    train_metrics["best_final_score"] = float(best_snapshot.get("final_score", 0))
    train_metrics["best_sensitivity"] = float(best_snapshot.get("sensitivity", 0))
    train_metrics["best_specificity"] = float(best_snapshot.get("specificity", 0))
    train_metrics["best_epoch"] = int(best_metric_epoch)
    train_metrics["total_epochs"] = epoch + 1
    
    return train_metrics


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train baseline model with 5-fold CV")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold to run (0-4). If None, runs all 5 folds.")
    parser.add_argument("--mode", type=str, default="2D", choices=["2D", "3D"], help="Model mode: 2D or 3D")
    args = parser.parse_args()
    
    # Create config with the specified mode
    config = Configuration(mode=args.mode)
    
    logging.info(f"🔧 Configuration: MODE={config.MODE}")
    
    # Setup GroupKFold directory
    groupkfold_dir = config.WORKDIR.parent / "luna25-baseline-public" / "dataset" / "luna25_csv_groupkfold"
    
    if not groupkfold_dir.exists():
        logging.error(f"❌ GroupKFold directory not found: {groupkfold_dir}")
        logging.error("Please run: python create_groupkfold_splits.py first")
        exit(1)
    
    # Determine which folds to run
    folds_to_run = [args.fold] if args.fold is not None else range(5)
    
    logging.info(f"\n{'='*80}")
    logging.info(f"🚀 LUNA25 BASELINE {config.MODE} MODEL - 5-FOLD CROSS-VALIDATION")
    logging.info(f"{'='*80}")
    logging.info(f"MODE: {config.MODE}")
    logging.info(f"Folds to run: {list(folds_to_run)}")
    logging.info(f"Dataset: {config.DATADIR}")
    logging.info(f"Results dir: {config.EXPERIMENT_DIR}")
    logging.info(f"{'='*80}\n")
    
    # Store results for averaging
    fold_results = {}
    all_metrics = []
    
    # Loop through folds
    for fold_num in folds_to_run:
        logging.info(f"\n{'='*80}")
        logging.info(f"🔄 FOLD {fold_num + 1}/5")
        logging.info(f"{'='*80}\n")
        
        # Load fold CSVs
        train_csv = groupkfold_dir / f"train_fold{fold_num}.csv"
        valid_csv = groupkfold_dir / f"valid_fold{fold_num}.csv"
        
        if not train_csv.exists() or not valid_csv.exists():
            logging.error(f"❌ Fold {fold_num} CSV files not found!")
            continue
        
        # Create experiment directory for this fold
        experiment_name = f"{config.EXPERIMENT_NAME}-{config.MODE}-fold{fold_num}-{datetime.today().strftime('%Y%m%d')}"
        exp_save_root = config.EXPERIMENT_DIR / experiment_name
        exp_save_root.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"📁 Results saved to: {exp_save_root}")
        
        # Train this fold
        try:
            fold_metrics = train(
                train_csv_path=train_csv,
                valid_csv_path=valid_csv,
                exp_save_root=exp_save_root,
                config=config,
            )
            fold_metrics["fold_number"] = fold_num
            fold_metrics["experiment_dir"] = str(exp_save_root)
            all_metrics.append(fold_metrics)
            
            # Save fold metrics to JSON
            fold_metrics_file = exp_save_root / "fold_metrics.json"
            with open(fold_metrics_file, 'w') as f:
                json.dump(fold_metrics, f, indent=2)
            logging.info(f"✅ Fold {fold_num + 1} completed successfully!")
            logging.info(f"   - Best AUC: {fold_metrics['best_auc']:.4f}")
            logging.info(f"   - Sensitivity: {fold_metrics['best_sensitivity']:.4f}")
            logging.info(f"   - Specificity: {fold_metrics['best_specificity']:.4f}")
            
        except Exception as e:
            logging.error(f"❌ Fold {fold_num + 1} failed with error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logging.info(f"\n{'='*80}")
    logging.info(f"🎉 ALL FOLDS COMPLETED!")
    logging.info(f"{'='*80}\n")
    
    # Generate summary report
    if all_metrics:
        summary_file = config.EXPERIMENT_DIR / f"{config.MODE}_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Calculate statistics for multiple metrics
        metric_keys = ['best_auc', 'best_final_score', 'best_sensitivity', 'best_specificity']
        stats = {}
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
            
        mean_epochs = float(np.mean([m['best_epoch'] for m in all_metrics]))
        
        summary = {
            "mode": config.MODE,
            "total_folds": len(all_metrics),
            "timestamp": datetime.now().isoformat(),
            "fold_results": all_metrics,
            "statistics": {
                "metrics": stats,
                "mean_epochs": mean_epochs,
                "config": {
                    "learning_rate": config.LEARNING_RATE,
                    "batch_size": config.BATCH_SIZE,
                    "epochs": config.EPOCHS,
                    "patience": config.PATIENCE,
                }
            }
        }
        
        # Save JSON summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV summary for easy viewing
        csv_file = config.EXPERIMENT_DIR / f"{config.MODE}_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_file, 'w', newline='') as f:
            # Update header with new columns
            fieldnames = ['fold_number', 'best_auc', 'best_final_score', 'best_sensitivity', 'best_specificity', 
                          'best_epoch', 'total_epochs', 'train_samples', 'valid_samples', 
                          'malignant_train', 'benign_train', 'malignant_valid', 'benign_valid']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metric in all_metrics:
                row = {k: v for k, v in metric.items() if k in fieldnames}
                # Format floats to 4 decimal places
                for key in ['best_auc', 'best_final_score', 'best_sensitivity', 'best_specificity']:
                    if key in row:
                        row[key] = f"{row[key]:.4f}"
                writer.writerow(row)
        
        logging.info(f"📊 Training Summary:")
        logging.info(f"   Mean AUC: {stats['best_auc']['mean']:.4f} ± {stats['best_auc']['std']:.4f}")
        logging.info(f"   Mean Sensitivity: {stats['best_sensitivity']['mean']:.4f}")
        logging.info(f"   Mean Specificity: {stats['best_specificity']['mean']:.4f}")
        logging.info(f"   Summary saved to: {summary_file}")
        logging.info(f"   CSV saved to: {csv_file}\n")