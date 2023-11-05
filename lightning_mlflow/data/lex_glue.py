import os
from typing import Optional

import polars as pl
import torch
from datasets import Dataset, DatasetDict, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LexGlueDataModule(LightningDataModule):
    """Customizes the setup and dataloader methods of the Lightning Data superclass
    to the lex_glue data. This data module is ultimately passed to the trainer.

    Attributes:
        pretrained_model: pass model name as a kwarg to preprocessor.
        max_length: token limit.
        batch_size: the same value for the DataLoader of all splits.
        num_workers: DataLoader arg that should equal number of cores.
        debug_mode_sample: takes a small sample to flush out runtime bugs.
        dsdict: a DatasetDict object with keys 'train', 'validation', and 'test';
            each of which is a Dataset object.
    """

    def __init__(
        self,
        pretrained_model: str,
        max_length: int,
        batch_size: int,
        num_workers: Optional[int] = 4,
        debug_mode_sample: Optional[int | None] = None,
    ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug_mode_sample = debug_mode_sample
        self.dsname = "lex_glue"
        self.dsdict = DatasetDict()

        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

    def prepare_data(self):
        load_dataset(self.dsname, "unfair_tos")

    def setup(self, stage: str = None) -> None:
        """Split dsdict into train, validation, test.

        Args:
            stage: `fit` or `test`
        """
        if stage == "fit" or stage is None:
            self.dsdict["train"] = load_dataset(
                self.dsname,
                "unfair_tos",
                split="train",
            )
            self.dsdict["validation"] = load_dataset(
                self.dsname,
                "unfair_tos",
                split="validation",
            )

        if stage == "test" or stage is None:
            self.dsdict["test"] = load_dataset(
                self.dsname,
                "unfair_tos",
                split="test",
            )

    def train_dataloader(self) -> DataLoader:
        """Use _shared_transform to preprocess the training split into a tensor,
        and pass this tensor into DataLoader.
        """
        return DataLoader(
            dataset=self._shared_transform("train"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Use _shared_transform to preprocess the validation split into a tensor,
        and pass this tensor into DataLoader.
        """
        return DataLoader(
            dataset=self._shared_transform("validation"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Use _shared_transform to preprocess the validation split into a tensor,
        and pass this tensor into DataLoader.
        """
        return DataLoader(
            dataset=self._shared_transform("test"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def _shared_transform(self, split: str) -> torch.tensor:
        """Tokenize the given split, and then convert from arrow to pytorch
        tensor format.
        """
        ds = self.dsdict[split]

        if self.debug_mode_sample is not None:
            df = pl.from_arrow(ds.data.table)
            df = _balanced_sample(df, self.debug_mode_sample)
            ds = Dataset.from_pandas(df.to_pandas())

        # Tokenize 'text' field to the 2 new fields: input_ids & attention_mask.
        # And also dichotomize the target ('label') to binary.
        tokenized_ds = ds.map(
            self._preprocess,
            batched=True,
            load_from_cache_file=True,
        )
        tokenized_ds.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        return tokenized_ds

    def _preprocess(self, batch: dict) -> dict:
        """Combines a tokenizer step with a target conversion step for binary
        classification. Input a batch of examples, tokenize the 'text' field;
        dichotomize the 'labels' field into 0 (fair) vs 1 (unfair).
        """
        tokens = self.tokenizer(
            batch["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        tokens["label"] = [1 if label else 0 for label in batch["labels"]]

        return tokens


def _balanced_sample(
    df: pl.DataFrame,
    sample_size: int,
    seed: int = 42,
) -> pl.DataFrame:
    """Equalize the sample sizes of the 2 classes to make a balanced dataframe
    and then take a random sample.
    """
    fairness = df["labels"].apply(lambda x: min(len(x), 1))  # Dichotomize code list

    fair = df.filter(fairness.eq(0)).sample(fairness.sum(), seed=seed)
    unfair = df.filter(fairness.ne(0))
    balanced = pl.concat([fair, unfair])

    return balanced.sample(n=sample_size, seed=seed)
