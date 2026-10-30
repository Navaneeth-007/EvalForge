"""Reproducible single-GPU experiment; run from the repository root on CUDA."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ARCHIVE_REPO = 'HAL-9001/spider-databases'
ARCHIVE_REVISION = '4a01bbac6520cd35b216db9e1724e5e1ada60aa4'
ARCHIVE_SHA256 = '00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--sample-size', type=int)
    parser.add_argument('--eval-n', type=int, default=200)
    args = parser.parse_args()
    import torch
    from huggingface_hub import hf_hub_download
    from train.dataset_prep import prepare_dataset
    from train.qlora_config import TRAINING_CONFIG
    from train.train import train

    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    archive = Path(hf_hub_download(ARCHIVE_REPO, 'spider_data.zip', repo_type='dataset', revision=ARCHIVE_REVISION))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError('Spider archive checksum mismatch')
    data_root = Path('data/spider_archive')
    data_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            target = (data_root / item.filename).resolve()
            if not target.is_relative_to(data_root.resolve()):
                raise ValueError('Unsafe archive member')
        source.extractall(data_root)
    tables = next(data_root.rglob('tables.json'))
    databases = tables.parent / 'database'
    if not databases.is_dir():
        raise ValueError('Archive is missing the Spider database directory')
    os.environ['SPIDER_DB_DIR'] = str(databases.resolve())
    prepare_dataset(tables_path=str(tables), sample_size=args.sample_size)
    TRAINING_CONFIG.update(num_train_epochs=args.epochs, per_device_train_batch_size=1,
                           per_device_eval_batch_size=1, gradient_accumulation_steps=16,
                           logging_steps=10, max_seq_length=2048)
    # The prepared data already contains the selected sample; never re-prepare inside train().
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    trainer = train(use_4bit=True)
    elapsed = time.monotonic() - started
    manifest_path = Path(TRAINING_CONFIG['output_dir']) / 'training_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(gpu=torch.cuda.get_device_name(0), elapsed_seconds=elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    archive_repo=ARCHIVE_REPO, archive_revision=ARCHIVE_REVISION,
                    archive_sha256=digest, tables_sha256=hashlib.sha256(tables.read_bytes()).hexdigest(),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    packages=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    from eval.run_eval import run_eval
    run_eval(base_model=TRAINING_CONFIG['model_name'], finetuned_model=None,
             adapter_dir=str(Path(TRAINING_CONFIG['output_dir']) / 'final'),
             eval_samples_path='data/processed/eval_samples.json', n=args.eval_n,
             output_path='outputs/eval_results.json', use_4bit=True, skip_bertscore=True)
    print('Experiment complete: preserve outputs/ and update the model card with measured evidence.')


if __name__ == '__main__':
    main()
