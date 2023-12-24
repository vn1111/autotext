import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


class TextDataset:
    def __init__(
            self,
            df: pd.DataFrame,
            text_col: str,
            target_col: str,
            train_set_size: float = 0.8,
            val_set_size: float = 0.1,
            test_set_size: float = 0.1
    ):
        # check test split values are valid
        TextDataset._valid_split_size(train_set_size, val_set_size, test_set_size)

        self._text_col = text_col
        self._target_col = target_col

        self._create_hf_dataset(df, train_set_size, val_set_size, test_set_size)

    @staticmethod
    def _valid_split_size(*args):
        """
        checker for the size of the train, validation and test sets. Value should be in the range [0, 1]
        """
        for split_size in args:
            if not 0 <= split_size <= 1:
                raise ValueError('train, validation and test sizes should all be between 0 and 1')
        if sum(args) != 1.:
            raise ValueError('train, validation and test size ration should sum up to 1')

    def _generate_train_val_test_sets(self, df, train_size, validation_size, test_size):
        """
        generate train, validation and test splits for the input dataset
        :param df: pd.DataFrame
            input df
        :param train_size: float
        :param validation_size: float
        :param test_size: float
        :return: tuple
            train, validation and test datasets in the form of pd dataframes with 2 columns; text and label
        """
        temp_df = df.copy()
        temp_df = temp_df.rename(columns={self._text_col: 'text', self._target_col: 'label'})
        ds_train, ds_val = train_test_split(
            temp_df,
            train_size=train_size,
            stratify=temp_df['label'],
            random_state=42
        )
        ds_val, ds_test = train_test_split(
            ds_val,
            train_size=validation_size/(validation_size + test_size),
            stratify=ds_val['label'],
            random_state=42
        )
        return ds_train, ds_val, ds_test

    def _create_hf_dataset(self, *args):
        """
        creating a datasets Dataset object so can be passed to Trainer easily
        """
        train, val, test = self._generate_train_val_test_sets(*args)
        self._dataset = DatasetDict({
            'train': Dataset.from_pandas(train.reset_index(drop=True)),
            'validation': Dataset.from_pandas(val.reset_index(drop=True)),
            'test': Dataset.from_pandas(test.reset_index(drop=True)),
        })

    @property
    def dataset(self):
        return self._dataset
