from .base import AbstractDataset

import pandas as pd

from datetime import date


class JsonDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'json'

    @classmethod
    def url(cls):
        return 'https://drive.google.com/drive/folders/1VrcwGR9m0aZyMk7mNNfd3BO9MT8tB02F?usp=sharing'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return [
            'item_names_table.json',
            'test_user_label.json',
            'train_user_seq_log.json',
            'test_user_seq_log.json'
        ]

    def load_df(self):
        folder_path = self._get_rawdata_folder_path()
        dfs = []
        for file in self.all_raw_file_names():
            file_path = folder_path.joinpath(file)
            df = pd.read_json(file_path, typ='table')
            df = df.to_frame(name='data')
            dfs.append(df)
        return dfs


