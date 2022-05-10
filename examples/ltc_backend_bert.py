# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
Runs a training of the Bert model using the Lazy Tensor Core with the
example Torch MLIR backend.

Most of the code in this example was copied from the wonderful tutorial
    https://huggingface.co/transformers/training.html#fine-tuning-in-native-pytorch

Based on LTC code samples by ramiro050
    https://github.com/ramiro050/lazy-tensor-samples
"""

import argparse
import torch
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, \
    BertTokenizer, AdamW, get_scheduler
from typing import List


def tokenize_dataset(dataset: DatasetDict) -> DatasetDict:
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')

    return tokenized_datasets


def train(model: BertForSequenceClassification,
          num_epochs: int,
          num_training_steps: int,
          train_dataloader: DataLoader,
          device: torch.device,
          do_mark_step: bool) -> List[torch.Tensor]:
    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler('linear', optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    model.train()
    losses = []
    for _ in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            losses.append(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if do_mark_step and 'lazy' in str(model.device):
                print("Calling Mark Step")
                torch._lazy.mark_step()

    return losses


def main(device, lower_only):
    if device in ("TS", "MLIR_EXAMPLE"):
        import torch._lazy

        if device == "TS":
            import torch._lazy.ts_backend

            torch._lazy.ts_backend.init()

        elif device == "MLIR_EXAMPLE":
            import ltc_backend.ltc_backend._EXAMPLE_MLIR_BACKEND as ltc_backend

            ltc_backend._initialize()

        device = "lazy"
        print("Initialized backend")
    else:
        device = device.lower()

    tokenized_datasets = tokenize_dataset(load_dataset('imdb'))
    small_train_dataset = tokenized_datasets['train'].shuffle(seed=42) \
        .select(range(2))

    train_dataloader = DataLoader(small_train_dataset, shuffle=True,
                                  batch_size=8)
    model = BertForSequenceClassification.from_pretrained('bert-base-cased',
                                                          num_labels=2)
    model.to(device)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    losses = train(model, num_epochs,
                   num_training_steps, train_dataloader, device, not lower_only)

    if lower_only:
        print('\nJIT Graph:')
        import torch._C
        graph_str = torch._C._lazy._get_tensors_backend([losses[0]])
        print(graph_str)
    else:
        # Execute computation
        print('Loss: ', losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        type=str.upper,
        choices=["CPU", "TS", "MLIR_EXAMPLE"],
        default="MLIR_EXAMPLE",
        help="The device type",
    )
    parser.add_argument(
        "-l",
        "--lower_only",
        action='store_true',
        default=False,
        help="Only get backend printout -- do not execute computation",
    )
    args = parser.parse_args()
    main(args.device, args.lower_only)
