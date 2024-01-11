from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from utils.supervised_metrics import compute_metrics
from ..dataset import TextDataset


class Classifier:
    def __init__(
            self,
            model_checkpoint: str,
            dataset: TextDataset

    ):
        self._base_training_args = TrainingArguments(
            output_dir=f'model_checkpoints_{model_checkpoint}',
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy='epoch',
            save_strategy='no',
            load_best_model_at_end=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=dataset.num_labels,
            id2label=dataset.id2label,
            label2id=dataset.label2id
        )

        self._trainer = trainer = Trainer(
            model=self._model,
            args=self._base_training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=self._tokenizer,
            compute_metrics=compute_metrics
        )

    def train(self, tune='no'):
        if tune == 'no':
            self._trainer.train()
        elif tune == 'quick':
            # quick hparam search
            pass
        elif tune == 'extensive':
            # extensive param search
            pass
        else:
            raise TypeError("tune should be one of ['no', 'quick', 'extensive']")
