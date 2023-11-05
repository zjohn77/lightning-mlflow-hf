"""
Takes a pretrained model with classification head and uses the peft package to do Adapter + LoRA
fine tuning.
"""
from typing import Any

import torch
from lightning import LightningModule
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW, Optimizer
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_f1_score,
    binary_precision,
    binary_recall,
)
from transformers import AutoModelForSequenceClassification


class TransformerModule(LightningModule):
    def __init__(
        self,
        pretrained_model: str,
        num_classes: int,
        lr: float,
    ):
        super().__init__()

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model,
            num_labels=num_classes,
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()

        self.lr = lr

        self.save_hyperparameters("pretrained_model")

    def forward(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        label: list[int],
    ):
        """Calc the loss by passing inputs to the model and comparing against ground
        truth labels. Here, all of the arguments of self.model comes from the
        SequenceClassification head from HuggingFace.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label,
        )

    def _compute_metrics(self, batch, split) -> tuple:
        """Helper method hosting the evaluation logic common to the <split>_step methods."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            label=batch["label"],
        )

        # For predicting probabilities, do softmax along last dimension (by row).
        prob_class1 = torch.argmax(torch.softmax(outputs["logits"], dim=-1), dim=1)

        metrics = {
            f"{split}_Loss": outputs["loss"],
            f"{split}_Acc": binary_accuracy(
                preds=prob_class1,
                target=batch["label"],
            ),
            f"{split}_F1_Score": binary_f1_score(
                preds=prob_class1,
                target=batch["label"],
            ),
            f"{split}_Precision": binary_precision(
                preds=prob_class1,
                target=batch["label"],
            ),
            f"{split}_Recall": binary_recall(
                preds=prob_class1,
                target=batch["label"],
            ),
        }

        return outputs, metrics

    def training_step(self, batch, batch_idx):
        outputs, metrics = self._compute_metrics(batch, "Train")
        self.log_dict(metrics, on_epoch=True, on_step=False)

        return outputs["loss"]

    def validation_step(self, batch, batch_idx) -> dict[str, Any]:
        _, metrics = self._compute_metrics(batch, "Val")
        self.log_dict(metrics, on_epoch=True, on_step=False)

        return metrics

    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        _, metrics = self._compute_metrics(batch, "Test")
        self.log_dict(metrics)

        return metrics

    def configure_optimizers(self) -> Optimizer:
        return AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=0.0,
        )
