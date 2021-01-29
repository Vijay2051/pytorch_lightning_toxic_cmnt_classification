import torch
from torch.utils.data import Dataset, DataLoader
import config


class ToxicCommentsDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len):
        self.data = data
        self.tokenizer = (tokenizer,)
        self.max_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        comment_text = data_row.comment_text
        labels = data_row[config.LABEL_COLUMNS]
        encoding = self.tokenizer[0].encode_plus(
            comment_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return dict(
            comment_text=comment_text,
            input_ids=encoding.input_ids.flatten(),
            attention_mask=encoding.attention_mask.flatten(),
            labels=torch.FloatTensor(labels),
        )


# train_dataset = ToxicCommentsDataset(config.df, config.TOKENIZER, max_token_len=128)
# print(train_dataset[0])