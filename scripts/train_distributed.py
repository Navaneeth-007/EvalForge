"""Training-only worker launched by scripts.run_gpu after shared data preparation."""
import argparse
import os


def main():
    import torch
    from train.qlora_config import TRAINING_CONFIG
    from train.train import train

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    TRAINING_CONFIG.update(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        max_seq_length=2048,
    )
    train(use_4bit=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
