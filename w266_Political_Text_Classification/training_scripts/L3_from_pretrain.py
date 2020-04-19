# SOURCES: We acknowledge and give thanks to the following technical resources, guides, and example scripts to inform the creation of this .py script
# Based off example https://github.com/huggingface/transformers/blob/master/examples/run_glue.py 
# Based off example https://aws.amazon.com/blogs/machine-learning/maximizing-nlp-model-performance-with-automatic-model-tuning-in-amazon-sagemaker/
# Based off example https://github.com/danwild/sagemaker-sentiment-analysis/blob/163913a21837683e7605f6122ad2c10718347f65/train/train.py#L45
# Based off example https://mccormickml.com/2019/07/22/BERT-fine-tuning/#3-tokenization--input-formatting


# GOAL: Use a pre-trained model and run the final Sequence classification task

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json
import time
import datetime
import pandas as pd
import collections

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, BertConfig,BertForSequenceClassification, BertTokenizer)
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import glue_processors as processors
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # see if this is the correct path
    model = BertForSequenceClassification.from_pretrained(
            model_info['model_name_or_path'], # 12-layer BERT model
            num_labels = model_info['num_labels'], # Number of output labels 
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False # Whether the model returns all hidden-states.
        )

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    return model

def str2bool(v): ## added for SageMaker run
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) 

def flat_f1(preds, labels):
    pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    f1 = f1_score(labels_flat, pred_flat, average = 'micro')
    return f1

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
        
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.tsv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

def train(model, args, train_dataset, train_dataloader, dev_dataset, dev_dataloader):
    print("Training Goes here")
    epochs = args.num_train_epochs
    optimizer = AdamW(model.parameters(),
                  lr = args.learning_rate , # args.learning_rate - default is 5e-5
                  eps = args.adam_epsilon # args.adam_epsilon  - default is 1e-8.
                )
    
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    # From https://mccormickml.com/2019/07/22/BERT-fine-tuning/#3-tokenization--input-formatting
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    # Store the average loss after each epoch so we can plot them.
    f1_values_train = []
    f1_values_dev = []
    loss_values = []
    
    # For each epoch...
    for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        
        # Measure how long the training epoch takes.
        t0 = time.time()
        
        # Reset the total loss for this epoch.
        total_loss = 0
        all_preds = np.array([])
        all_labels = np.array([])

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the `to` method.
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(args.device)
            b_input_mask = batch[1].to(args.device)
            b_labels = batch[2].to(args.device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]
            logits = outputs[1]
            
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value from the tensor.
            total_loss += loss.item()
            
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            
            # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            
            # Update the learning rate.
            scheduler.step()

            # Get Metrics
            label_ids = b_labels.to('cpu').numpy()
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis = 1)
            all_preds = np.concatenate((all_preds, preds), axis = None)
            all_labels = np.concatenate((all_labels, label_ids), axis = None)
            

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        f1 = flat_f1(all_preds, all_labels)  
        f1_values_train.append(f1)         
    
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  F1: {0:.2f}".format(f1))
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
        
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differentlyduring evaluation.
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_preds = np.array([])
        all_labels = np.array([])
        
        # Evaluate data for one epoch
        for batch in dev_dataloader:
            
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
        
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
        
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():        
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            preds = np.argmax(logits, axis = 1)
            all_preds = np.concatenate((all_preds, preds), axis = None)
            all_labels = np.concatenate((all_labels, label_ids), axis = None)
        
            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            tmp_n = batch[0].size()[0]
        
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches and examples
            nb_eval_steps += 1
            nb_eval_examples += tmp_n

        # Report the final accuracy for this validation run.
        # print(all_preds)
        # print(all_labels)
        # print(f1(all_preds,all_labels))
        f1 = flat_f1(all_preds, all_labels)
        f1_values_dev.append(f1)
        print("  F1: {0:.2f}".format(f1))
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_examples))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("F1 Train:", f1_values_train)
    print("F1 Dev:", f1_values_dev)
    print("Training complete!")


def evaluate(candidate, args, model):
    print("Evaluate on Test Dataset!")
    examples = pd.read_csv(os.path.join(args.data_dir, '%s.tsv'%candidate), sep = "\t", header = 0, lineterminator='\n')
    print("Num records:",examples.shape[0])
    
    all_input_ids = examples.id.to_list()
    print('Tokenize testing, like training')
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
    tokenized_input = examples.text.apply(lambda x: tokenizer.encode(x,add_special_tokens = True, max_length = 128, pad_to_max_length = True))
    
    attn_masks = []
    for sent in tokenized_input:
        att_mask = [int(token > 0) for token in sent]
        attn_masks.append(att_mask)
    
    # Convert to Tensors
    all_input_ids = torch.tensor(all_input_ids).to(args.device)
    attn_masks = torch.tensor(attn_masks).to(args.device)
    tokenized_input = torch.tensor(tokenized_input).to(args.device)
    
    test_dataset = TensorDataset(tokenized_input, attn_masks)
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)
    print("Created test dataset and loader")
    
    # Now evaluate
    model.to(args.device)
    model.eval()
    predictions = []
    # Predict 
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(args.device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
        b_input_ids = b_input_ids.to(args.device)
        b_input_mask = b_input_mask.to(args.device)
        
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis = 1)
        
        # Store predictions and true labels
        predictions.extend(list(preds))
    print("All predictions made")
        
    # Now calculate the stance on various issues
    agg = collections.Counter(predictions)
    
    try:
        immigration_stance = (((agg[0] - agg[1]) * 1.0 / (agg[0] + agg[1])) + 1) * 0.5
    except:
        immigration_stance = 0.5
    print("immigration_stance", immigration_stance)    
    try:
        gun_stance = (((agg[2] - agg[3]) * 1.0 / (agg[2] + agg[3])) + 1) * 0.5
    except:
        gun_stance = 0.5
    print("gun_stance", gun_stance)
    try:
        medicare_stance = (((agg[4] - agg[5]) * 1.0 / (agg[4] + agg[5])) + 1) * 0.5
    except:
        medicare_stance = 0.5
    print("medicare_stance",medicare_stance)
    try:
        abortion_stance = (((agg[6] - agg[7]) * 1.0 / (agg[6] + agg[7])) + 1) * 0.5
    except:
        abortion_stance = 0.5
    print("abortion_stance",abortion_stance)
    try:
        free_college_stance = (((agg[8] - agg[9]) * 1.0 / (agg[8] + agg[9])) + 1) * 0.5
    except:
        free_college_stance = 0.5
    print("free_college_stance",free_college_stance)
    try:
        spending_stance = (((agg[10] - agg[11]) * 1.0 / (agg[10] + agg[11])) + 1) * 0.5
    except:
        spending_stance = 0.5
    print("spending_stance",spending_stance)
    try:
        wealth_tax_stance = (((agg[12] - agg[13]) * 1.0 / (agg[12] + agg[13])) + 1) * 0.5
    except:
        wealth_tax_stance = 0.5
    print("wealth_tax_stance",wealth_tax_stance)


    print('Total Predictions:', len(predictions))
    
    immigration_importance = (agg[0] + agg[1]) * 1.0 / len(predictions)
    guns_importance = (agg[2] + agg[3]) * 1.0 / len(predictions)
    medicare_importance = (agg[4] + agg[5]) * 1.0 / len(predictions)
    abortion_importance = (agg[6] + agg[7]) * 1.0 / len(predictions)
    college_importance = (agg[8] + agg[9]) * 1.0 / len(predictions)
    spending_importance = (agg[10] + agg[11]) * 1.0 / len(predictions)
    tax_importance = (agg[12] + agg[13]) * 1.0 / len(predictions)
    
    print(candidate, ': stance on immigration:', immigration_stance, ".... Relative Importance:", immigration_importance)
    print(candidate, ':stance on guns:', gun_stance, ".... Relative Importance:", guns_importance)
    print(candidate, ': stance on medicare:', medicare_stance, ".... Relative Importance:", medicare_importance)
    print(candidate, ': stance on abortion:', abortion_stance, ".... Relative Importance:", abortion_importance)
    print(candidate, ': stance on free college:', free_college_stance, ".... Relative Importance:", college_importance)
    print(candidate, ': stance on military spending:', spending_stance, ".... Relative Importance:", spending_importance)
    print(candidate, ': stance on wealth tax:', wealth_tax_stance, ".... Relative Importance:", tax_importance)

def load_and_cache_examples(args, dataset):
    print("Preprocessing Goes here")
    processor = processors["mrpc"]()
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()
    if dataset == 'dev':
        examples = pd.read_csv(os.path.join(args.data_dir, 'dev.tsv'), sep = "\t", header = 0)
    else:
        examples = pd.read_csv(os.path.join(args.data_dir, 'train.tsv'), sep = "\t", header = 0) 
    print('Directory:', args.data_dir)
    print(examples.head())
    
    
    labels = examples.label.to_list()
    all_input_ids = examples.id.to_list()
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
    
    # Tokenizing the input using BertTokenizer. Setting a max_length of 128.
    tokenized_input = examples.text.apply(lambda x: tokenizer.encode(x,add_special_tokens = True, max_length = 128, pad_to_max_length = True))
    
    print("Creating Attention Masks")
    attn_masks = []
    for sent in tokenized_input:
        att_mask = [int(token > 0) for token in sent]
        attn_masks.append(att_mask)
        
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
 
    # Convert to Tensors
    all_input_ids = torch.tensor(all_input_ids)
    attn_masks = torch.tensor(attn_masks)
    tokenized_input = torch.tensor(tokenized_input)
    labels = torch.tensor(labels)
    
    dataset = TensorDataset(tokenized_input, attn_masks, labels)
    return dataset

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", type=str,required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", required=True, type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: BERT only")
#     parser.add_argument("--task_name", default=None, type=str, required=True,
#                         help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_train_epochs", default=None, type=int, required=True,
                        help="The number of epochs for training.")
    parser.add_argument("--num_labels", default=None, type=int, required=True,
                        help="The number of unique labels (2 for binary classification).")

    # Other Parameters
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--seed', type=int, default=101,
                        help="random seed for initialization")
    parser.add_argument("--do_train", type=str2bool, nargs='?', const=True, default=False, ## modified for SageMaker use
                        help="Whether to run training.")
    parser.add_argument("--do_eval", type=str2bool, nargs='?', const=True, default=False, ## modified for SageMaker use
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", type=str2bool, nargs='?', const=True, default=False, ## modified for SageMaker use
                        help="Avoid using CUDA when available")
    parser.add_argument('--fp16', type=str2bool, nargs='?', const=True, default=False, ## modified for SageMaker use
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    args = parser.parse_args()
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Training
    if args.do_train:
        batch_size = 32
        
        train_dataset = load_and_cache_examples(args, dataset='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
        print("Created train dataset and loader")
        
        dev_dataset = load_and_cache_examples(args, dataset='dev')
        dev_sampler = RandomSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=batch_size)                
        print("Created dev dataset and loader")
        
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path = args.model_name_or_path, # 12-layer BERT model
            num_labels = args.num_labels, # Number of output labels 
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        
        model.to(args.device)
        logger.info("Training/evaluation parameters %s", args)    
            
        train(model, args, train_dataset, train_dataloader, dev_dataset, dev_dataloader)
        
        # Save the parameters used to construct the model
        model_info_path = os.path.join(args.model_dir, 'model_info.pth')
        with open(model_info_path, 'wb') as f:
            model_info = {'model_name_or_path': args.model_name_or_path,
                         'num_labels': args.num_labels
                         }
            torch.save(model_info, f)
        
        # Save the model parameters
        model_path = os.path.join(args.model_dir, 'model.pth')
        with open(model_path, 'wb') as f:
            torch.save(model.cpu().state_dict(), f)       
        print("saved model path")
        
        evaluate("Biden", args, model) 
        evaluate("Sanders", args, model)
        evaluate("Warren", args, model)
        evaluate("Yang", args, model)
        evaluate("Buttigieg", args, model)
        evaluate("Klobuchar", args, model)
        evaluate("Trump", args, model)
     

if __name__ == "__main__":
    main()
    

