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
import sys
from typing import List

import torch
import torch._C
import torch._lazy
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification,
    BertConfig,
    BertTokenizer,
    AdamW,
    get_scheduler,
)


def tokenize_dataset(dataset: DatasetDict) -> DatasetDict:
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    return tokenized_datasets


def train(
    model: BertForSequenceClassification,
    num_epochs: int,
    num_training_steps: int,
    train_dataloader: DataLoader,
    device: torch.device,
) -> List[torch.Tensor]:
    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

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

            if "lazy" in str(model.device):
                print("Calling Mark Step")
                torch._lazy.mark_step()

    return losses


def main(device="lazy", full_size=False):
    """
    Load model to specified device. Ensure that any backends have been initialized by this point.

    :param device: name of device to load tensors to
    :param full_size: if true, use a full pretrained bert-base-cased model instead of a smaller variant
    """
    torch.manual_seed(0)

    tokenized_datasets = tokenize_dataset(load_dataset("imdb"))
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2))

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    if full_size:
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=2
        )
    else:
        configuration = BertConfig(
            vocab_size=28996,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=512,
            layer_norm_eps=1.0e-05,
        )
        model = BertForSequenceClassification(configuration)

    model.to(device)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    losses = train(model, num_epochs, num_training_steps, train_dataloader, device)

    # Get debug information from LTC
    if "torch_mlir._mlir_libs._REFERENCE_LAZY_BACKEND" in sys.modules:
        computation = lazy_backend.get_latest_computation()
        if computation:
            print(computation.debug_string())

    print("Loss: ", losses)

    return model, losses


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
        "-f",
        "--full_size",
        action="store_true",
        default=False,
        help="Use full sized BERT model instead of one with smaller parameterization",
    )
    args = parser.parse_args()

    if args.device in ("TS", "MLIR_EXAMPLE"):
        if args.device == "TS":
            import torch._lazy.ts_backend

            torch._lazy.ts_backend.init()

        elif args.device == "MLIR_EXAMPLE":
            import torch_mlir._mlir_libs._REFERENCE_LAZY_BACKEND as lazy_backend

            lazy_backend._initialize()

        device = "lazy"
        print("Initialized backend")
    else:
        device = args.device.lower()

    main(device, args.full_size)
