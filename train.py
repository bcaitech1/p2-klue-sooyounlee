import random
import argparse
import glob
import random
import re
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from transformers import Trainer, TrainingArguments, XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification

from load_data import *


# seed 고정
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


def increment_output_dir(output_path, exist_ok=False):
    path = Path(output_path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(args):
#    wandb.init(project='P_Stage2', entity='leesooyoun',
#    tags=args.tag_list, group=args.group, name=args.run_name)

    seed_everything(args.seed)

    # load model and tokenizer
    MODEL_NAME = args.pretrained_model
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    # split dataset
    dataset = pd.read_csv('/opt/ml/input/data/train/train.tsv', delimiter='\t', header=None)
    train, dev = train_test_split(dataset, test_size=0.2, random_state=42)
    train.to_csv('/opt/ml/input/data/train/train_train.tsv', sep='\t', header=None, index=False)
    dev.to_csv('/opt/ml/input/data/train/train_dev.tsv', sep='\t', header=None, index=False)

    # load dataset
    train_dataset = load_data("/opt/ml/input/data/train/train_train.tsv")

    dev_dataset = load_data("/opt/ml/input/data/train/train_dev.tsv")

    train_label = train_dataset['label'].values
    dev_label = dev_dataset['label'].values

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setting model hyperparameter
    model_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 42
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model.to(device)

    output_dir = increment_output_dir(args.output_dir)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        save_total_limit=args.save_total_limit,  # number of total save model.
        save_steps=args.save_steps,  # model saving step.
        num_train_epochs=args.epochs,  # total number of training epochs
        learning_rate=args.lr,  # learning_rate
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,  # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=100,            # evaluation step.
        dataloader_num_workers=4,
#        label_smoothing_factor=0.5
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )

    # train model
    trainer.train()


def main(args):
    train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_type', type=str, default='XMLRoberta')
    parser.add_argument('--pretrained_model', type=str, default='xlm-roberta-large')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=500)  # number of warmup steps for learning rate scheduler
    parser.add_argument('--output_dir', type=str, default='./results/expr')
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--logging_dir', type=str, default='./logs')  # directory for storing logs


    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)