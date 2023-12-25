import unittest
from autotext.dataset import TextDataset
import pandas as pd
from datasets import DatasetDict, Dataset


class TestTextDataset(unittest.TestCase):
    def test_split_sizes(self):
        txt = 10 * ['a'] + 20 * ['b']
        y = 10 * [1] + 20 * [0]
        df = pd.DataFrame({'text': txt, 'y': y})
        with self.assertRaises(ValueError):
            ds = TextDataset(
                df,
                'text',
                'y',
                1.1,
                .1,
                .1
            )
            ds = TextDataset(
                df,
                'text',
                'y',
                .8,
                1.1,
                .1
            )
            ds = TextDataset(
                df,
                'text',
                'y',
                .1,
                .1,
                .1
            )

    def test_dataset(self):
        txt = 10 * ['a'] + 20 * ['b']
        y = 10 * [1] + 20 * [0]
        df = pd.DataFrame({'text': txt, 'y': y})
        ds = TextDataset(
            df,
            'text',
            'y',
            .8,
            .1,
            .1
        ).dataset
        # check entire dataset is a dataset dict
        self.assertIsInstance(ds, DatasetDict)
        # check splits are Dataset objects
        self.assertIsInstance(ds['train'], Dataset)
        # check splits sizes
        self.assertEqual(len(ds), 30)
        self.assertEqual(len(ds['train']), 24)
        self.assertEqual(len(ds['validation']), 3)
        self.assertEqual(len(ds['test']), 3)


if __name__ == '__main__':
    unittest.main()
