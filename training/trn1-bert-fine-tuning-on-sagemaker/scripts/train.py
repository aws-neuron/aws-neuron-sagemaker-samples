from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import logging
import sys
import argparse
import os
import tarfile

# Set Neuron Persistent Cache’s root directory to SM_OUTPUT_DATA_DIR (/opt/ml/output/data/) so it will be uploaded to S3 at the end of the training job.
os.environ['NEURON_CC_FLAGS'] = '--cache_dir=/opt/ml/output/data/'

# Enable torchrun
import torch
import torch_xla.distributed.xla_backend
if os.environ.get("WORLD_SIZE"):
    torch.distributed.init_process_group("xla")

# Fixup to enable distributed training with XLA
orig_wrap_model = Trainer._wrap_model
def _wrap_model(self, model, training=True):
    self.args.local_rank = -1
    return orig_wrap_model(self, model, training)
Trainer._wrap_model = _wrap_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--learning_rate', type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])   
    parser.add_argument('--training_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test_dir', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--s3_prefix_dir', type=str, default=os.environ['SM_CHANNEL_S3_PREFIX'])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName('INFO'),
        handlers=[logging.StreamHandler(sys.stdout)],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Load and extract Neuron Persistent Cache to SM_OUTPUT_DATA_DIR
    with tarfile.open(f'{args.s3_prefix_dir}/neuron-compile-cache.tar.gz', 'r') as t:
        t.extractall(path=args.output_data_dir)

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f' loaded train_dataset length is: {len(train_dataset)}')
    logger.info(f' loaded test_dataset length is: {len(test_dataset)}')

    # compute metrics function for binary classification.
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=float(args.learning_rate),
        save_total_limit=1,
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # train model
    trainer.train()
    
    # evaluate model
    eval_result = trainer.evaluate()

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, 'eval_results.txt'), 'w') as writer:
        print(f'***** Eval results *****')
        for key, value in sorted(eval_result.items()):
            writer.write(f'{key} = {value}\n')
    
    # Saves the model to s3
    trainer.save_model(args.model_dir)