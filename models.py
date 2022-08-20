import torch
import os
import numpy as np

from transformers import GPT2LMHeadModel, AutoConfig, AutoModelWithLMHead, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.autograd import Variable


class CausalLM:


    def __init__(self, model_name, lr, tokenizer, device):
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.resize_token_embeddings(len(tokenizer))
        self.optimizer = AdamW(self.model.parameters(), lr=lr, eps=1e-8)
        self.lr = lr
        self.tokenizer = tokenizer
        self.device = device


    def reset_optimizers(self, lr=None):
        '''
        Reinitializes `self.optimizer`.
        '''
        self.optimizer = AdamW(self.model.parameters(), lr=lr if lr is not None else self.lr, eps=1e-8)
        
            
    def save_model_and_optimizers(self, info_dict, save_path):
        '''
        Saves `self.model.state_dict()`, along with any additional information 
        in `info_dict`, into `save_path`.
        
        Also saves the state of optimizer:
        `self.optimizer.state_dict()`.
        '''
        # place model and optimizer state dicts into info_dict
        info_dict['model_state_dict'] = self.model.state_dict()
        info_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # save
        torch.save(info_dict, save_path)
    
    
    def load_model(self, checkpoint_root, experiment_name, checkpoint_name):
        '''
        Load the model and optimizers from f'{checkpoint_root}/{experiment_name}/{checkpoint_name}'.
        
        If checkpoint does not contain optimizer state dicts, the optimizers are reinitialized
        using `self.reset_optimizers()`
        '''
        checkpoint_path = os.path.join(checkpoint_root, experiment_name, checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint.keys():
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print('no optimizer state dict found in checkpoint; resetting optimizers...')
            self.reset_optimizers()


    def step_basic(self, batch):
        '''
        Performs a standard (min. negative log likelihood / max. log likelihood)
        gradient update on `self.model` using `batch`.
        
        Returns:
            1. `loss`: the loss of this batch before updating.
        '''

        self.optimizer.zero_grad()

        # max sequence length
        seq_len = batch['input_ids'].shape[1]
        
        # each tensor to device
        input_ids       = batch['input_ids'].to(self.device)
        target_ids      = batch['target_ids'].to(self.device)
        attention_mask  = batch['attention_masks'].to(self.device)
        # print(input_ids.shape, position_ids.shape, token_type_ids.shape, target_ids.shape, attention_mask.shape)

        # get model outputs
        outputs = self.model(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            labels = target_ids
        )
        loss = outputs.loss

        # # calculate the negative loss
        # loss = CausalLM._calculate_mle_loss(
        #     logits = outputs.logits,  # (batch_size, seq_len)
        #     labels = target_ids,  # (batch_size, seq_len, vocab_size)
        #     ignore_index = -100   # TODO: remove this magic number
        # )
        loss.backward()
        
        # consider gradient clipping here before updating
        
        self.optimizer.step()
        
        return  {
            'loss': loss
        }


    def step_calc_loss_manually(self, batch):
        '''
        Performs a standard (min. negative log likelihood / max. log likelihood)
        gradient update on `self.model` using `batch`.
        
        Returns:
            1. `loss`: the loss of this batch before updating.
        '''

        self.optimizer.zero_grad()

        # max sequence length
        seq_len = batch['input_ids'].shape[1]
        
        # each tensor to device
        input_ids       = batch['input_ids'].to(self.device)
        target_ids      = batch['target_ids'].to(self.device)
        attention_mask  = batch['attention_masks'].to(self.device)
        # print(input_ids.shape, position_ids.shape, token_type_ids.shape, target_ids.shape, attention_mask.shape)

        # get model outputs
        outputs = self.model(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            labels = target_ids
        )

        # calculate the negative loss
        loss = CausalLM._calculate_mle_loss(
            logits = outputs.logits,  # (batch_size, seq_len)
            labels = target_ids,  # (batch_size, seq_len, vocab_size)
            ignore_index = -100   # TODO: remove this magic number
        )
        loss.backward()
        
        # consider gradient clipping here before updating
        
        self.optimizer.step()
        
        return  {
            'loss': loss
        }

    
    def nll_on_dataset(self, dataloader):
        '''
        Returns :
            :param nll: float; the average negative log-likelihood on the examples in a `dataloader`.
        '''
        self.model.eval()
        
        batch_wise_losses = []
        len_loader = len(dataloader)
        for batch_id, batch in enumerate(dataloader):

            if batch_id % 10 == 0:
                print(f'\rEvaluating batch: {batch_id} of {len_loader}', end='', flush=True)

            # calculate negative log likelihood loss
            # each tensor to device
            input_ids       = batch['input_ids'].to(self.device)
            target_ids      = batch['target_ids'].to(self.device)
            attention_mask  = batch['attention_masks'].to(self.device)
            # print(input_ids.shape, position_ids.shape, token_type_ids.shape, target_ids.shape, attention_mask.shape)

            # get model outputs
            outputs = self.model(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                labels = target_ids
            )
            batch_wise_losses.append(outputs.loss.item())
            
        return np.array(batch_wise_losses).mean()


    @staticmethod
    def _calculate_mle_loss(logits, labels, ignore_index):
        '''
        Returns the (unweighted) negative log likelihood of a batch.
        This is the usual loss function we use in language modeling.
        '''
        
        # get masked logprobs
        masked_logprobs_of_true_labels, ignore_index_mask = \
        CausalLM._calculate_masked_logprobs(
            logits, labels, ignore_index
        )
        
        # loss
        example_wise_loss = torch.sum(masked_logprobs_of_true_labels, dim = 1)
        loss = - example_wise_loss.sum() / ignore_index_mask.sum()
        return loss

    
    @staticmethod
    def _calculate_masked_logprobs(logits, labels, ignore_index):
        '''
        Calculates the (masked) log-probabilities of `labels` under the model's `logits`.
        Locations in `labels` with value `ignore_index` will be masked (will have logprob zero).
        
        Inputs:
            :param `logits`: Tensor (batch_size, seq_len, vocab_size); 
                the logits the model calculated given some inputs.
            :param `labels`: Tensor (batch_size, seq_len);
                the true token ids whose logprobs you are interested.
            :param `ignore_index`: int;
                locations in `labels` with this value will be masked (will have logprob zero).
                
        Returns:
            :param `masked_logprobs`: masked log-probabilities of `labels` from `logits`.
            :param `ignore_index_mask`: mask used to mask away `ignore_index` values in `labels.
        '''
        batch_size = logits.shape[0]
        seq_len    = logits.shape[1]
        vocab_size = logits.shape[-1]

        # shift labels to align with logits
        shift_labels = labels[..., 1:].contiguous()

        # and truncate logits to make shapes the same
        trunc_logits = logits[..., :-1, :].contiguous()

        # flatten logits and labels
        flat_logits = trunc_logits.view(-1, trunc_logits.shape[-1])  # (batch_size*seq_len, vocab_size)
        flat_labels = shift_labels.view(-1)  # (batch_size*seq_len)

        # calculate the log probabilities
        flat_logprobs = torch.nn.functional.log_softmax(flat_logits, dim = -1)
        
        # for each example, grab the log probability corresponding to the true label
        onehot_labels = NegativePositiveTrainingLM._onehot_maskgen(flat_logprobs.size(), flat_labels.data).to(torch.bool)
        logprobs_of_true_labels = torch.masked_select(flat_logprobs, onehot_labels).view(batch_size, -1)  # (batch_size, seq_len)

        # mask out locations with value `ignore_index`
        ignore_index_mask = ~(shift_labels == ignore_index)  # 0 when value == ignore_index, 1 otherwise
        return logprobs_of_true_labels * ignore_index_mask, ignore_index_mask



class NegativePositiveTrainingLM:

    
    def __init__(
        self, 
        model_name,
        model_config, 
        pos_lr, neg_lr, 
        num_opt_steps, 
        tokenizer, 
        device, 
        opt_eps = 1e-8, 
        sched_weight_decay = 0.0,
        sched_num_warmup_steps = 0
    ):
    
        self.model = GPT2LMHeadModel.from_pretrained(model_name, config=model_config).to(device)
        # self.model = AutoModelWithLMHead.from_pretrained(model_name, config=model_config).to(device)
        self.model_config = model_config
        self.pos_lr = pos_lr
        self.neg_lr = neg_lr
        self.tokenizer = tokenizer
        self.device = device
        self.num_opt_steps = num_opt_steps
        self.opt_eps = opt_eps
        self.sched_weight_decay = sched_weight_decay
        self.sched_num_warmup_steps = sched_num_warmup_steps
        self.init_optimizers_and_schedulers()
        self.curr_accum_steps = 0


    def init_optimizers_and_schedulers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        self.no_decay = ["bias", "LayerNorm.weight"]
        self.optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)],
                "weight_decay": self.sched_weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 
                "weight_decay": 0.0
            },
        ]
        self.optimizer_pos = AdamW(self.optimizer_grouped_parameters, lr=self.pos_lr, eps=self.opt_eps)
        self.optimizer_neg = AdamW(self.optimizer_grouped_parameters, lr=self.neg_lr, eps=self.opt_eps)
        self.scheduler_pos = get_linear_schedule_with_warmup(
            self.optimizer_pos, 
            num_warmup_steps = self.sched_num_warmup_steps, 
            num_training_steps = self.num_opt_steps
        )
        self.scheduler_neg = get_linear_schedule_with_warmup(
            self.optimizer_neg, 
            num_warmup_steps = self.sched_num_warmup_steps, 
            num_training_steps = self.num_opt_steps
        )
        
        
    def reset_optimizers(self):
        '''
        Reinitializes `self.optimizer_pos` and `self.optimizer_neg`.
        '''
        self.init_optimizers_and_schedulers()

            
    def save_model_and_optimizers(self, info_dict, save_path):
        '''
        Saves `self.model.state_dict()`, along with any additional information 
        in `info_dict`, into `save_path`.
        
        Also saves the state of both optimizers:
        `self.optimizer_pos.state_dict()` and `self.optimizer_neg.state_dict()`.
        '''
        # place model and optimizer state dicts into info_dict
        info_dict['model_state_dict'] = self.model.state_dict()
        info_dict['optimizer_pos_state_dict'] = self.optimizer_pos.state_dict()
        info_dict['optimizer_neg_state_dict'] = self.optimizer_neg.state_dict()

        # TODO: save schedulers as well
        
        # save
        torch.save(info_dict, save_path)
    
    
    def load_model(self, checkpoint_root, experiment_name, checkpoint_name):
        '''
        Load the model and optimizers from f'{checkpoint_root}/{experiment_name}/{checkpoint_name}'.
        
        If checkpoint does not contain optimizer state dicts, the optimizers are reinitialized
        using `self.reset_optimizers()`
        '''
        checkpoint_path = os.path.join(checkpoint_root, experiment_name, checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_pos_state_dict' in checkpoint.keys() and 'optimizer_neg_state_dict' in checkpoint.keys():
            self.optimizer_pos.load_state_dict(checkpoint['optimizer_pos_state_dict'])
            self.optimizer_neg.load_state_dict(checkpoint['optimizer_neg_state_dict'])
            # TODO: load schedulers as well
        else:
            print('no optimizer state dict found in checkpoint; resetting optimizers...')
            self.reset_optimizers()


    def positive_step_basic(self, batch, max_grad_norm=1.0, debug=False):
        '''
        Performs a standard (min. negative log likelihood / max. log likelihood)
        gradient update on `self.model` using `batch`.
        
        Returns:
            1. `loss`: the loss of this batch before updating.
        '''

        # self.optimizer_pos.zero_grad()

        # max sequence length
        seq_len = batch['input_ids'].shape[1]
        
        # each tensor to device
        input_ids       = batch['input_ids'].to(self.device)
        position_ids    = batch['position_ids'].to(self.device)
        token_type_ids  = batch['token_type_ids'].to(self.device)
        target_ids      = batch['target_ids'].to(self.device)
        attention_mask  = batch['attention_masks'].to(self.device)
        # print(input_ids.shape, position_ids.shape, token_type_ids.shape, target_ids.shape, attention_mask.shape)

        # get model outputs
        self.model.train()  # this changes the outputs somehow!!
        outputs = self.model(
            input_ids = input_ids, 
            # attention_mask = attention_mask,
            # token_type_ids = token_type_ids,
            # position_ids = position_ids,
            labels = target_ids
        )
        loss = outputs.loss
        if debug:
            print('example input ids: ', input_ids[0], input_ids[0].shape)
            print('example target ids: ', target_ids[0], target_ids[0].shape)
            print('example inputs: ', self.tokenizer.decode(input_ids[0]))
            print('example targets:', self.tokenizer.decode(target_ids[0]))
            print('model output logits:', outputs.logits)
            print('model loss:', loss)

        # # calculate the negative loss
        # loss = CausalLM._calculate_mle_loss(
        #     logits = outputs.logits,  # (batch_size, seq_len)
        #     labels = target_ids,  # (batch_size, seq_len, vocab_size)
        #     ignore_index = -100   # TODO: remove this magic number
        # )
        loss.backward()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        self.optimizer_pos.step()
        self.scheduler_pos.step()
        self.model.zero_grad()
        
        return  {
            'loss': loss
        }

    
    def positive_step(self, batch, max_grad_norm=1.0, grad_accum_steps=64, debug=False):
        '''
        Performs a standard (min. negative log likelihood / max. log likelihood)
        gradient update on `self.model` using `batch`.
        
        Returns:
            1. `loss`: the loss of this batch before updating.
        '''

        # self.optimizer_pos.zero_grad()

        # max sequence length
        seq_len = batch['input_ids'].shape[1]
        
        # each tensor to device
        input_ids       = batch['input_ids'].to(self.device)
        position_ids    = batch['position_ids'].to(self.device)
        token_type_ids  = batch['token_type_ids'].to(self.device)
        target_ids      = batch['target_ids'].to(self.device)
        attention_mask  = batch['attention_masks'].to(self.device)
        # print(input_ids.shape, position_ids.shape, token_type_ids.shape, target_ids.shape, attention_mask.shape)

        # get model outputs
        self.model.train()  # Best practices - ALWAYS call this just before making gradient updates!
        outputs = self.model(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            labels = target_ids
        )

        # calculate the negative loss
        loss = NegativePositiveTrainingLM._calculate_positive_phase_loss(
            logits = outputs.logits,  # (batch_size, seq_len)
            labels = target_ids,  # (batch_size, seq_len, vocab_size)
            ignore_index = -100   # TODO: remove this magic number
        )
        if debug:
            print('example input ids: ', input_ids[0], input_ids[0].shape)
            print('example target ids: ', target_ids[0], target_ids[0].shape)
            print('example inputs: ', self.tokenizer.decode(input_ids[0]))
            print('example targets:', self.tokenizer.decode(target_ids[0]))
            print('model output logits:', outputs.logits)
            print('model loss:', loss)

        loss.backward()

        if self.curr_accum_steps == grad_accum_steps:

            print(f'\rMaking gradient update...', end='', flush=True)

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.optimizer_pos.step()
            self.scheduler_pos.step()
            self.model.zero_grad()

            self.curr_accum_steps = 0
        else:
            self.curr_accum_steps += 1
        
        return  {
            'loss': loss
        }
    
    
    def negative_step(
        self, batch, 
        max_grad_norm = 1.0,
        EXAMPLE_WEIGHT_MODE = 'decay',
        EXAMPLE_WEIGHT_CARE_MODE = 'sample_min', 
        EXAMPLE_WEIGHT_REJECTION_THRESHOLD = -5.0
    ):
        '''
        Performs a negative (max. negative log likelihood / min. log likelihood)
        gradient update on `self.model` using negative examples `batch`.
        
        `batch` should contain negative examples ONLY; i.e. the examples you want 
        `self.model` to avoid generating.
        
        Inputs:
            :param batch: negative examples you want to update the model with. 
            For the other params, see docs of `NegativePositiveTrainingLM._calculate_negative_phase_loss_new`.
            
        Returns:
            :param `loss`: weighted loss of `batch` before updating. This loss is the special loss; see `NegativePositiveTrainingLM._calculate_negative_phase_loss_new`.
            :param `unweighted_loss`: unweighted loss of `batch` before updating.
            :param `example_weights`: weights used to weight the loss; one weight per example in `batch`.
            :param `average_nll_true_labels`: the negative log-likelihood (the usual loss metric) of the labels.  
        '''

        # self.optimizer_neg.zero_grad()

        # max sequence length
        seq_len = batch['input_ids'].shape[1]
        
        # each tensor to device
        input_ids       = batch['input_ids'].to(self.device)
        position_ids    = batch['position_ids'].to(self.device)
        token_type_ids  = batch['token_type_ids'].to(self.device)
        target_ids      = batch['target_ids'].to(self.device)
        attention_mask  = batch['attention_masks'].to(self.device)
        # print(input_ids.shape, position_ids.shape, token_type_ids.shape, target_ids.shape, attention_mask.shape)

        # get model outputs
        self.model.train()  # Best practices - ALWAYS call this just before making gradient updates!
        outputs = self.model(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            labels = target_ids
        )
        
        # calculate the negative loss
        loss, unweighted_loss, example_weights, average_nll_true_labels = NegativePositiveTrainingLM._calculate_negative_phase_loss_new(
            logits = outputs.logits,  # (batch_size, seq_len)
            labels = target_ids,  # (batch_size, seq_len, vocab_size)
            ignore_index = -100,  # TODO: remove this magic number
            WEIGHT_MODE = EXAMPLE_WEIGHT_MODE,
            CARE_MODE = EXAMPLE_WEIGHT_CARE_MODE,
            REJECTION_THRESHOLD = EXAMPLE_WEIGHT_REJECTION_THRESHOLD,
        )
        loss.backward()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        self.optimizer_pos.step()
        self.scheduler_pos.step()
        self.model.zero_grad()
        
        return {
            'loss': loss,
            'unweighted_loss': unweighted_loss,
            'example_weights': example_weights,
            'average_nll_true_labels': average_nll_true_labels,
        }
    
    
    def forward_logprobs_of_batch(self, batch):
        '''
        Returns the forward pass results of `batch` in their (masked) log-prob form.
        Masking is performed on the basis of `ignore_index` (TODO: change 
        `ignore_index` from a magic number to a proper parameter); all tokens with
        value `ignore_index` will contribute a log probability of zero.
        
        Inputs:
            :param `batch`: the batch for which you want to calculate the logprobs.
        
        Returns:
            :param `masked_logprobs`: (masked) log-probabilities of `batch` under the model. 
                Tokens with value `ignore_index` will have log-probability zero.
            :param `ignore_index_mask`: the mask used to mask away the original log-probabilities.
        '''

        # max sequence length
        seq_len = batch['input_ids'].shape[1]
        
        # each tensor to device
        with torch.no_grad():
            input_ids       = batch['input_ids'].to(self.device)
            position_ids    = batch['position_ids'].to(self.device)
            token_type_ids  = batch['token_type_ids'].to(self.device)
            attention_mask  = batch['attention_masks'].to(self.device)
            target_ids      = batch['target_ids'].to(self.device)
            # print(input_ids.shape, position_ids.shape, token_type_ids.shape, target_ids.shape, attention_mask.shape)

            # get model outputs
            outputs = self.model(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                position_ids = position_ids,
                labels = target_ids
            )
            
            # calculate the masked logprobs
            masked_logprobs_of_true_labels, ignore_index_mask = \
            NegativePositiveTrainingLM._calculate_masked_logprobs(
                logits = outputs.logits,  # (batch_size, seq_len)
                labels = target_ids,  # (batch_size, seq_len, vocab_size)
                ignore_index = -100,  # TODO: remove this magic number
            )
            
        return  {
            'masked_logprobs': masked_logprobs_of_true_labels,  # (batch_size, seq_len)
            'ignore_index_mask': ignore_index_mask  # (batch_size, seq_len)
        }


    def perplexity_on_dataset_basic(self, dataloader):
        self.model.eval()
        batch_wise_losses = []
        eval_loss = 0.0
        nb_eval_steps = 0
        len_loader = len(dataloader)

        for batch_id, batch in enumerate(dataloader):
            
            if batch_id % 10 == 0:
                print(f'\rEvaluating batch: {batch_id} of {len_loader}', end='', flush=True)

            # each tensor to device
            input_ids       = batch['input_ids'].to(self.device)
            target_ids      = batch['target_ids'].to(self.device)
            
            with torch.no_grad():
                # get model outputs
                outputs = self.model(
                    input_ids = input_ids,
                    labels = target_ids
                )
                lm_loss = outputs[0]
                # print(lm_loss)
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        result = {"perplexity": perplexity}
            
        return perplexity


    def perplexity_on_dataset(self, dataloader):
        self.model.eval()
        batch_wise_losses = []
        eval_loss = 0.0
        nb_eval_steps = 0
        len_loader = len(dataloader)

        for batch_id, batch in enumerate(dataloader):
            
            if batch_id % 10 == 0:
                print(f'\rEvaluating batch: {batch_id} of {len_loader}', end='', flush=True)

            # each tensor to device
            input_ids       = batch['input_ids'].to(self.device)
            target_ids      = batch['target_ids'].to(self.device)
            
            with torch.no_grad():
                # get model outputs
                outputs = self.model(
                    input_ids = input_ids,
                    labels = target_ids
                )
                lm_loss = NegativePositiveTrainingLM._calculate_positive_phase_loss(
                    logits = outputs.logits,  # (batch_size, seq_len)
                    labels = target_ids,  # (batch_size, seq_len, vocab_size)
                    ignore_index = -100   # TODO: remove this magic number
                )
                # print(lm_loss)
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        result = {"perplexity": perplexity}
            
        return perplexity
    
    
    def nll_on_dataset(self, dataloader):
        '''
        Returns :
            :param nll: float; the average negative log-likelihood on the examples in a `dataloader`.
        '''
        self.model.eval()
        
        batch_wise_losses = []
        len_loader = len(dataloader)
        for batch_id, batch in enumerate(dataloader):

            if batch_id % 10 == 0:
                print(f'\rEvaluating batch: {batch_id} of {len_loader}', end='', flush=True)

            # calculate negative log likelihood loss
            logprobs_result = self.forward_logprobs_of_batch(batch)
            example_wise_loss = torch.sum(logprobs_result['masked_logprobs'], dim = 1)
            loss = -example_wise_loss.sum() / logprobs_result['ignore_index_mask'].sum()
            batch_wise_losses.append(loss.item())
            
        return np.array(batch_wise_losses).mean()
    
    
    def generate_response(self, inputs, max_new_tokens=100):
        '''
        Given a single prompt, generate a reponse.
       
        Inputs:
            :param `inputs`: dict[Tensor]; must contain at least `input_ids` and `attention_mask` 
                corresponding to one single example.
                
        Returns:
            :param `output_ids`: numerical representation of model outputs.
            :param `decoded_outputs`: decoded outputs using `self.tokenizer`.
        '''
        
        # each tensor to device
        input_ids       = inputs['input_ids'].to(self.device)
        attention_mask  = inputs['attention_masks'].to(self.device)
        
        # ensure only one input is given
        assert input_ids.shape[0] == 1
        assert attention_mask.shape[0] == 1
        
        outputs = self.model.generate(
            inputs = input_ids, 
            max_new_tokens = max_new_tokens,
            pad_token_id = self.tokenizer.eos_token_id,
            do_sample = True,  # do_sample = True; otherwise all questions will be identical
            top_p = 0.95,      # nucleus sampling
            top_k = 0,         # deactivate top-k words sampling
            attention_mask = attention_mask  # do not pay attention to padding
        ).cpu()
        
        decoded_outputs = [self.tokenizer.decode(output) for output in outputs]
        
        return {
            'output_ids': outputs, 
            'decoded_outputs': decoded_outputs
        }
    
    
    def batch_generate_responses(self, batch, max_new_tokens=100, return_ground_truth=False):
        '''
        Given a collection of prompts, generate a response to each.
        
        Inputs:
            :param batch: dictionary of 2-dimensional torch Tensors, 
                containing at least `input_ids` and `attention_mask`.
                Each of these tensors must be of shape (batch_size, seq_len).
            
        Returns:
            :param `output_ids`: numerical representation of model outputs.
            :param `decoded_outputs`: decoded outputs using `self.tokenizer`.
            :param `decoded_ground_truth_outputs`: if `target_ids` in `batch`,
                this output contains the ground truth label (response).
        '''
        
        self.model.eval()
        
        # max sequence length
        seq_len = batch['input_ids'].shape[1]
        
        # each tensor to device
        input_ids       = batch['input_ids'].to(self.device)
        # position_ids    = batch['position_ids'].to(self.device)
        # token_type_ids  = batch['token_type_ids'].to(self.device)
        # target_ids      = batch['target_ids'].to(self.device)
        attention_mask  = batch['attention_masks'].to(self.device)
        
        outputs = self.model.generate(
            inputs = input_ids,  # BUG: we are actually feeding in the ground truth responses here.
            max_new_tokens = max_new_tokens,
            pad_token_id = self.tokenizer.eos_token_id,
            do_sample = True,  # do_sample = True; otherwise all questions will be identical
            top_p = 0.95,      # nucleus sampling
            top_k = 0,         # deactivate top-k words sampling
            attention_mask = attention_mask  # do not pay attention to padding
        ).cpu()
        
        decoded_outputs = [self.tokenizer.decode(output) for output in outputs]
        
        # if targets available, return as well
        target_ids = batch.get('target_ids', None)
        if return_ground_truth and target_ids is not None:
            target_ids      = target_ids.to(self.device)
            token_type_ids  = batch['token_type_ids'].to(self.device)
            
            decoded_ground_truth_outputs = []
            for example_id in range(target_ids.shape[0]):
                example = target_ids[example_id]
                example_mask = token_type_ids[example_id]
                target_ids_masked = torch.masked_select(example, example_mask > 0)
                decoded_ground_truth_outputs.append(self.tokenizer.decode(target_ids_masked.cpu()))
            
            return {
                'output_ids': outputs, 
                'decoded_outputs': decoded_outputs,
                'decoded_ground_truth_outputs': decoded_ground_truth_outputs
            }
        else:
            return {
                'output_ids': outputs, 
                'decoded_outputs': decoded_outputs
            }
    
        
    @staticmethod
    def _onehot_maskgen(sz, idx):
        msk = torch.BoolTensor(sz)
        msk.fill_(False)
        msk[torch.LongTensor(range(sz[0])), idx.cpu()] = True
        if idx.is_cuda == True:
            msk = msk.cuda()
        return Variable(msk)
        
    
    @staticmethod
    def _calculate_masked_logprobs(logits, labels, ignore_index):
        '''
        Calculates the (masked) log-probabilities of `labels` under the model's `logits`.
        Locations in `labels` with value `ignore_index` will be masked (will have logprob zero).
        
        Inputs:
            :param `logits`: Tensor (batch_size, seq_len, vocab_size); 
                the logits the model calculated given some inputs.
            :param `labels`: Tensor (batch_size, seq_len);
                the true token ids whose logprobs you are interested.
            :param `ignore_index`: int;
                locations in `labels` with this value will be masked (will have logprob zero).
                
        Returns:
            :param `masked_logprobs`: masked log-probabilities of `labels` from `logits`.
            :param `ignore_index_mask`: mask used to mask away `ignore_index` values in `labels.
        '''
        batch_size = logits.shape[0]
        seq_len    = logits.shape[1]
        vocab_size = logits.shape[-1]

        # shift labels to align with logits
        shift_labels = labels[..., 1:].contiguous()

        # and truncate logits to make shapes the same
        trunc_logits = logits[..., :-1, :].contiguous()

        # flatten logits and labels
        flat_logits = trunc_logits.view(-1, trunc_logits.shape[-1])  # (batch_size*seq_len, vocab_size)
        flat_labels = shift_labels.view(-1)  # (batch_size*seq_len)

        # calculate the log probabilities
        flat_logprobs = torch.nn.functional.log_softmax(flat_logits, dim = -1)
        
        # for each example, grab the log probability corresponding to the true label
        onehot_labels = NegativePositiveTrainingLM._onehot_maskgen(flat_logprobs.size(), flat_labels.data).to(torch.bool)
        logprobs_of_true_labels = torch.masked_select(flat_logprobs, onehot_labels).view(batch_size, -1)  # (batch_size, seq_len)

        # mask out locations with value `ignore_index`
        ignore_index_mask = ~(shift_labels == ignore_index)  # 0 when value == ignore_index, 1 otherwise
        return logprobs_of_true_labels * ignore_index_mask, ignore_index_mask
    
    
    @staticmethod
    def _calculate_masked_probs(logits, labels, ignore_index):
        '''
        Similar to `NegativePositiveTrainingLM._calculate_masked_logprobs`, but returns
        proper probabilities instead of log-probabilities.
        '''
        batch_size = logits.shape[0]
        seq_len    = logits.shape[1]
        vocab_size = logits.shape[-1]

        # shift labels to align with logits
        shift_labels = labels[..., 1:].contiguous()

        # and truncate logits to make shapes the same
        trunc_logits = logits[..., :-1, :].contiguous()

        # flatten logits and labels
        flat_logits = trunc_logits.view(-1, trunc_logits.shape[-1])  # (batch_size*seq_len, vocab_size)
        flat_labels = shift_labels.view(-1)  # (batch_size*seq_len)

        # calculate the log probabilities
        flat_logprobs = torch.nn.functional.softmax(flat_logits, dim = -1)
        
        # for each example, grab the log probability corresponding to the true label
        onehot_labels = NegativePositiveTrainingLM._onehot_maskgen(flat_logprobs.size(), flat_labels.data).to(torch.bool)
        logprobs_of_true_labels = torch.masked_select(flat_logprobs, onehot_labels).view(batch_size, -1)  # (batch_size, seq_len)

        # mask out locations with value `ignore_index`
        ignore_index_mask = ~(shift_labels == ignore_index)  # 0 when value == ignore_index, 1 otherwise
        return logprobs_of_true_labels * ignore_index_mask, ignore_index_mask
    
    
    @staticmethod
    def _calculate_negative_phase_loss_new(
        logits, labels, ignore_index,
        WEIGHT_MODE = 'decay',  # 'clamp', 'decay', 'none'
        CARE_MODE = 'sample_min', 
        REJECTION_THRESHOLD = -5.0,
    ):
        '''
        Calculates the loss for the negative phase, which maximizes the likelihood of
        NOT generating the true token in the negative batch (i.e. maximize log[1-p(true_token)])
        
        :param WEIGHT_MODE:
            1. 'decay': if an example in the negative batch is no longer 
                likely under the model, it will not contribute to the total loss
                (example_wise_loss[i] == 0 if i not likely)
            2. 'clamp': the loss of a token is clamped to a minimum of 7.5,
                s.t. the model will not try to optimize it further if it is already sufficiently
                unlikely.
            3. 'none': places no restrictions on the loss
        :param CARE_MODE: see _get_exmample_wise_loss_weights() method for details
        :param REJECTION_THRESHOLD: see _get_exmample_wise_loss_weights() method for details
        '''
        
        # get masked probs
        masked_probs_of_true_labels, ignore_index_mask = \
        NegativePositiveTrainingLM._calculate_masked_probs(
            logits, labels, ignore_index
        )
        
        # we are interested in the complement probability
        # i.e. the probability mass assigned to tokens that are NOT the true label.
        complement_probs = (1 - masked_probs_of_true_labels)
        
        # convert to log-probs
        complement_logprobs_masked = torch.log(complement_probs) * ignore_index_mask
        # print(complement_logprobs_masked)
        
        # get masked logprobs to evaluate the likelihood of the original, true token
        masked_logprobs_of_true_labels, ignore_index_mask = \
        NegativePositiveTrainingLM._calculate_masked_logprobs(
            logits, labels, ignore_index
        )
        # print(masked_logprobs_of_true_labels)
        
        # get example-wise loss weights (indicator for which examples are still likely)
        example_wise_loss_weights = NegativePositiveTrainingLM._get_exmample_wise_loss_weights(
            batch_logprobs = masked_logprobs_of_true_labels, 
            mask = ignore_index_mask, 
            CARE_MODE = CARE_MODE,  # 'sample_min'
            REJECTION_THRESHOLD = REJECTION_THRESHOLD  # -5.0
        )
        # print(example_wise_loss_weights)

        # loss
        example_wise_loss = torch.sum(complement_logprobs_masked, dim = 1)
        unweighted_loss = - example_wise_loss.sum() / ignore_index_mask.sum()
        # print(unweighted_loss)
        
        # average negative log-likelihood on true tokens
        example_wise_nll_true_labels = torch.sum(masked_logprobs_of_true_labels, dim = 1)
        average_nll_true_labels = - example_wise_nll_true_labels.sum() / ignore_index_mask.sum()
        
        return unweighted_loss, unweighted_loss, example_wise_loss_weights, average_nll_true_labels

    
    @staticmethod
    def _calculate_negative_phase_loss(
        logits, labels, ignore_index,
        WEIGHT_MODE = 'decay',  # 'clamp', 'decay', 'none'
        CARE_MODE = 'sample_min', 
        REJECTION_THRESHOLD = -5.0,
    ):
        '''
        Calculates the loss for the negative phase, weighted depending on how unlikely the
        negative example is under the model.
        
        
        :param WEIGHT_MODE:
            1. 'decay': if an example in the negative batch is no longer 
                likely under the model, it will not contribute to the total loss
                (example_wise_loss[i] == 0 if i not likely)
            2. 'clamp': the loss of a token is clamped to a minimum of 7.5,
                s.t. the model will not try to optimize it further if it is already sufficiently
                unlikely.
            3. 'none': places no restrictions on the loss
        :param CARE_MODE: see _get_exmample_wise_loss_weights() method for details
        :param REJECTION_THRESHOLD: see _get_exmample_wise_loss_weights() method for details
        '''
        
        # get masked logprobs
        masked_logprobs_of_true_labels, ignore_index_mask = \
        NegativePositiveTrainingLM._calculate_masked_logprobs(
            logits, labels, ignore_index
        )
        
        if WEIGHT_MODE == 'decay':
            # get example-wise loss weights for each negative example in batch
            example_wise_loss_weights = NegativePositiveTrainingLM._get_exmample_wise_loss_weights(
                batch_logprobs = masked_logprobs_of_true_labels, 
                mask = ignore_index_mask, 
                CARE_MODE = CARE_MODE,  # 'sample_min'
                REJECTION_THRESHOLD = REJECTION_THRESHOLD  # -5.0
            )

            # loss
            example_wise_loss = torch.sum(masked_logprobs_of_true_labels, dim = 1) 
            example_wise_loss_weighted = example_wise_loss * example_wise_loss_weights
            
            # usually we negate the sign of the log probabilities s.t. minimizing the loss maximizes the log probabilities
            # but here we don't negate it, s.t. minimizing the loss MINIMIZES the log probabilities of this batch
            loss            = example_wise_loss_weighted.sum() / ignore_index_mask.sum()
            unweighted_loss = example_wise_loss.sum() / ignore_index_mask.sum()
            return loss, unweighted_loss, example_wise_loss_weights
        
        
        elif WEIGHT_MODE == 'clamp':
            clamp_value = REJECTION_THRESHOLD - 0.5
            
            # get example-wise loss weights for each negative example in batch
            example_wise_loss_weights = NegativePositiveTrainingLM._get_exmample_wise_loss_weights(
                batch_logprobs = masked_logprobs_of_true_labels, 
                mask = ignore_index_mask, 
                CARE_MODE = CARE_MODE,  # 'sample_min'
                REJECTION_THRESHOLD = REJECTION_THRESHOLD  # -5.0
            )
            
            # loss: with value clamping
            batch_logprobs_clamped = torch.clamp(masked_logprobs_of_true_labels, min=clamp_value)
            example_wise_loss = torch.sum(batch_logprobs_clamped, dim = 1)
            # print(batch_logprobs_clamped)
            # print(example_wise_loss_weights)
            
            loss            = example_wise_loss.sum() / ignore_index_mask.sum()
            unweighted_loss = example_wise_loss.sum() / ignore_index_mask.sum()
            return loss, unweighted_loss, example_wise_loss_weights
        
        elif WEIGHT_MODE == 'none':
            # loss
            example_wise_loss = torch.sum(masked_logprobs_of_true_labels, dim = 1) 
            
            # usually we negate the sign of the log probabilities s.t. minimizing the loss maximizes the log probabilities
            # but here we don't negate it, s.t. minimizing the loss MINIMIZES the log probabilities of this batch
            loss            = example_wise_loss.sum() / ignore_index_mask.sum()
            unweighted_loss = example_wise_loss.sum() / ignore_index_mask.sum()
            return loss, unweighted_loss, None
        

    @staticmethod
    def _calculate_positive_phase_loss(logits, labels, ignore_index):
        '''
        Returns the (unweighted) negative log likelihood of a batch.
        This is the usual loss function we use in language modeling.
        '''
        
        # get masked logprobs
        masked_logprobs_of_true_labels, ignore_index_mask = \
        NegativePositiveTrainingLM._calculate_masked_logprobs(
            logits, labels, ignore_index
        )
        
        # loss
        example_wise_loss = torch.sum(masked_logprobs_of_true_labels, dim = 1)
        loss = - example_wise_loss.sum() / ignore_index_mask.sum()
        return loss
    
    
    @staticmethod
    def _get_exmample_wise_loss_weights(batch_logprobs, mask, CARE_MODE, REJECTION_THRESHOLD):
        '''
        Since the negative training phase theoretically has no bounds on how unlikely a data point can be 
        (the log-likelihood objective can reach minus infinity), we want to stop optimizing for 
        those datapoints which are already "sufficiently unlikely", i.e. exceeding some threshold. 

        The lower the rejection threshold is, the more unlikely an example needs to be for the loss to ignore it.

        Returns `example_wise_loss_weights`, assigns either `1` or `0` to each example in a negative batch:
            `1` indicates that the example is still considered likely under the model (> REJECTION_THRESHOLD)
            `0` indicates that the example is considered unlikely under the model (<= REJECTION_THRESHOLD)

        TODO: enable dynamic devices
        '''

        assert(CARE_MODE == 'sample_min' or CARE_MODE == 'sample_avg')
        batch_size = batch_logprobs.shape[0]
        example_wise_loss_weights = [0.0 for _ in range(batch_size)]

        # overwrite `example_wise_loss_weights` if an example is still likely
        for i in range(batch_size):

            filt_logprobs = torch.masked_select(batch_logprobs[i], mask[i])  # filter based on mask

            # if even one token in this sequence is still likely, this example should still contribute to the loss
            if CARE_MODE == 'sample_min' and torch.min(filt_logprobs).item() > REJECTION_THRESHOLD:
                example_wise_loss_weights[i] = 1
            
            # if the average token likelihoods in this sequence is still likely, example should still contribute to loss.
            # this gives us a measure of how likely the whole sequence is, not just the worst case token in that sequence.
            # seq_likelihood = sum(logprobs_in_seq) / len(seq)
            if CARE_MODE == 'sample_avg' and torch.sum(filt_logprobs).item() / len(filt_logprobs) > REJECTION_THRESHOLD:
                example_wise_loss_weights[i] = 1

        return torch.FloatTensor(example_wise_loss_weights).cuda()
