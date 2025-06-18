from dataloader import get_fold_dataloader
import traceback
import sys

def test_loader():
    try:
        train_loader = get_fold_dataloader(
            fold=0,
            split='train',
            batch_size=32,
            shuffle=False,
            pin_memory=True,
            preload_ram=True
        )
        val_loader = get_fold_dataloader(
            fold=0,
            split='val',
            batch_size=32,
            shuffle=False,
            pin_memory=True,
            preload_ram=True
        )

        print("Testing train_loader...")
        for i, (patch, mask) in enumerate(train_loader):
            print(f"Batch {i}: patch shape {patch.shape}, dtype {patch.dtype}")
            print(f"Batch {i}: mask shape {mask.shape}, dtype {mask.dtype}")
            print(f"Batch {i}: unique mask values {mask.unique()}")
            if i == 1:
                break

        print("\nTesting val_loader...")
        for i, (patch, mask) in enumerate(val_loader):
            print(f"Batch {i}: patch shape {patch.shape}, dtype {patch.dtype}")
            print(f"Batch {i}: mask shape {mask.shape}, dtype {mask.dtype}")
            print(f"Batch {i}: unique mask values {mask.unique()}")
            if i == 1:
                break
    except Exception as e:
        print(f"[ERROR] Exception in test_loader: {e}", file=sys.stderr)
        traceback.print_exc()

if __name__ == "__main__":
    test_loader()