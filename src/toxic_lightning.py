import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_lightning.metrics.functional.classification import auroc
from transformers import AdamW, get_linear_schedule_with_warmup, BertModel

from dataset import ToxicCommentsDataset
from config import BERT_MODEL, LABEL_COLUMNS, df, TOKENIZER, BATCH_SIZE, N_EPOCHS

"""
    : Lightning DataModule
"""


class ToxicCommentsDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, tokenizer, max_length=128, batch_size=8):
        super(ToxicCommentsDataModule, self).__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.batch_size = batch_size

    def setup(self):
        self.train_dataset = ToxicCommentsDataset(
            self.train_df, self.tokenizer, self.max_len
        )

        self.val_dataset = ToxicCommentsDataset(
            self.train_df, self.tokenizer, self.max_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=True, num_workers=4)


# initiate the setup in the datamodule
train_df, val_df = train_test_split(df, test_size=0.05)
train_toxic = train_df[train_df[LABEL_COLUMNS].sum(axis=1) > 0]
train_clean = train_df[train_df[LABEL_COLUMNS].sum(axis=1) == 0]
train_df = pd.concat([train_toxic, train_clean.sample(15_000)])
datamodule = ToxicCommentsDataModule(
    train_df, val_df, TOKENIZER, max_length=128, batch_size=BATCH_SIZE
)
datamodule.setup()


"""
    : Lightning Module
"""


class ToxicCommentClassifier(pl.LightningModule):
    def __init__(self, n_classes, steps_per_epoch, n_epochs):
        super(ToxicCommentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.classifier = (self.bert.config.hidden_size, n_classes)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
            return loss, output
        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": output, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []

        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)

        for output in outputs:
            for out_preds in output["predictions"].detach().cpu():
                predictions.append(out_preds)

        labels = torch.stack(labels)
        predictions = torch.stack(predictions)

        for i, name in enumerate(LABEL_COLUMNS):
            roc_score = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(
                f"{name}_roc_auc/Train", roc_score, self.current_epoch
            )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        return [optimizer], [scheduler]


# MODEL instatitated

model = ToxicCommentClassifier(
    n_classes=6, steps_per_epoch=len(train_df) // BATCH_SIZE, n_epochs=N_EPOCHS
)
trainer = pl.Trainer(max_epochs=N_EPOCHS, gpus=1, progress_bar_refresh_rate=20)
trainer.fit(model, datamodule)
