# Training log for checkpoints/best_model.pth

Trained on CPU (2 cores), EfficientNet-B0 (timm, RA weights), 224 px.

Leave-one-fruit-out:
    python3 src/train.py --cv --head-epochs 3 --finetune-epochs 4 --samples-per-epoch 2000 --val-cap 400
      apple 0.888 | banana 0.939 | orange 0.896 | strawberry 0.908 | mean 0.908

Final model (all four fruits, stratified 10% val):
    python3 src/train.py --head-epochs 3 --finetune-epochs 3 --samples-per-epoch 2500 --val-cap 150 --temperature 0.55
      val acc 0.991 (seen fruits), fitted T on seen fruits 0.32 -> saved T 0.55 (measured on unseen fruits)

    python3 src/inference.py samples/   -> 11/11 correct
