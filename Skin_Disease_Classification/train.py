import os
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime

from model import build_model
from data_loader import load_directory_data

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR    = r'os.environ.get('DATA_DIR', './data')'
IMG_SIZE    = (300, 300)
BATCH_SIZE  = 8
GRAD_ACCUM  = 2  # Effective batch size = BATCH_SIZE * GRAD_ACCUM = 16

PHASE1_EPOCHS       = 10
PHASE1_LR           = 1e-3

PHASE2_EPOCHS       = 60
PHASE2_LR           = 5e-5
FINE_TUNE_LAYERS    = 4  # Unfreeze top 4 blocks of efficientnet features out of 9

PHASE3_EPOCHS       = 30
PHASE3_LR           = 1e-5

# Checkpoint files
BEST_MODEL_PATH     = 'best_model.pth'
PHASE1_CKPT_PATH    = 'ckpt_phase1.pth'
PHASE2_CKPT_PATH    = 'ckpt_phase2.pth'
PHASE3_CKPT_PATH    = 'ckpt_phase3.pth'
PROGRESS_FILE       = 'training_progress_pt.json'

def save_progress(phase_done):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({'phase_done': phase_done}, f)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f).get('phase_done', 0)
    return 0

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    total_steps = len(loader)
    
    optimizer.zero_grad(set_to_none=True)
    
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            # Scale loss by accumulation steps so gradients scale correctly
            loss = criterion(outputs, labels) / GRAD_ACCUM
            
        scaler.scale(loss).backward()
        
        if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == total_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Multiply back by GRAD_ACCUM to record the true loss magnitude
        running_loss += loss.item() * GRAD_ACCUM * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if (i+1) % 50 == 0 or (i+1) == total_steps:
            print(f"  Step [{i+1}/{total_steps}] - Loss: {loss.item() * GRAD_ACCUM:.4f} - Acc: {correct/total:.4f}")
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    torch.cuda.empty_cache() # Clear fragmented VRAM at the end of epoch
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def run_phase(phase, model, train_loader, val_loader, device, class_weights, 
              epochs, lr, ckpt_path, label_smoothing=0.0, patience=5):
              
    print(f"\n" + "="*60)
    print(f"  STARTING PHASE {phase} - Epochs: {epochs}, LR: {lr}")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=lr, weight_decay=1e-4)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=patience//2, 
                                  verbose=True, min_lr=1e-8)
                                  
    scaler = torch.amp.GradScaler('cuda')
                                  
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    # Check if we can resume mid-phase
    start_epoch = 1
    if os.path.exists(ckpt_path):
        print(f"[RESUME] Loading checkpoint {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        if 'scaler_state' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"[RESUME] Resuming from epoch {start_epoch} with Best Val Acc: {best_val_acc:.4f}")

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        # Save per-epoch checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_val_acc': best_val_acc
        }, ckpt_path)
        
        # Save best global model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  [*] Best model saved! (Val Acc: {best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"  [EARLY STOPPING] Validation accuracy did not improve for {patience} epochs.")
            break
            
    # Return best accuracy and model
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    return best_val_acc

if __name__ == '__main__':
    print("=" * 60)
    print("  Skin Disease Prediction -- PyTorch GPU Migration")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device targeting: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    phase_done = load_progress()
    if phase_done > 0:
        print(f"\n[INFO] Resuming PyTorch training. Phase {phase_done} already complete.\n")

    train_loader, val_loader, num_classes, class_weights = load_directory_data(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )
    
    # Phase 1: Feature Extraction
    if phase_done < 1:
        model = build_model(num_classes=num_classes, fine_tune=False).to(device)
        acc = run_phase(1, model, train_loader, val_loader, device, class_weights, 
                        epochs=PHASE1_EPOCHS, lr=PHASE1_LR, 
                        ckpt_path=PHASE1_CKPT_PATH, patience=5)
        print(f"\n[DONE] Phase 1 complete. Best val_accuracy: {acc:.4f}")
        save_progress(1)
        
    # Phase 2: Partial Fine-Tuning
    if phase_done < 2:
        model = build_model(num_classes=num_classes, fine_tune=True, 
                            fine_tune_layers=FINE_TUNE_LAYERS).to(device)
        
        # Load best weights from previous runs if checkpoint doesn't exist
        if not os.path.exists(PHASE2_CKPT_PATH) and os.path.exists(BEST_MODEL_PATH):
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
            
        acc = run_phase(2, model, train_loader, val_loader, device, class_weights, 
                        epochs=PHASE2_EPOCHS, lr=PHASE2_LR, 
                        ckpt_path=PHASE2_CKPT_PATH, patience=12)
        print(f"\n[DONE] Phase 2 complete. Best val_accuracy: {acc:.4f}")
        save_progress(2)
        
    # Phase 3: Full Fine-Tuning
    if phase_done < 3:
        model = build_model(num_classes=num_classes, fine_tune=True, fine_tune_layers=999).to(device)
        
        if not os.path.exists(PHASE3_CKPT_PATH) and os.path.exists(BEST_MODEL_PATH):
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
            
        acc = run_phase(3, model, train_loader, val_loader, device, class_weights, 
                        epochs=PHASE3_EPOCHS, lr=PHASE3_LR, 
                        ckpt_path=PHASE3_CKPT_PATH, label_smoothing=0.1, patience=15)
        print(f"\n[DONE] Phase 3 complete. Best val_accuracy: {acc:.4f}")
        save_progress(3)
        
    print("\n[COMPLETE] PyTorch Training finished! Best model locally saved.", BEST_MODEL_PATH)