import torch
from torch.nn.utils.rnn import pad_sequence

class DialogCollate:
    
    '''
    Masks prompt tokens in the target tensor, so they do not contribute to the loss.
    
    collate_fn in dataloader is used for post processing on a single batch, 
    unlike __getitem__ in dataset class which returns a single example.
    '''
    
    def __init__(
        self, 
        tokenizer,
        max_len = 512,
        _targets_ignore_index = -100,
        _pad_token_id = 0
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len  # long sequences will be truncated to this length
        self._targets_ignore_index = _targets_ignore_index
        self._pad_token_id = _pad_token_id
        
    
    def _construct_input_ids(self, example):
        '''
        :param example: a dict consisting of 'src' and 'tgt' keys,
            corresponding to source (prompt) and response (target) utterances.
        '''

        if example['tgt']:
            example_input_ids = \
                example['src'] + [self.tokenizer.eos_token_id] + example['tgt'] + [self._pad_token_id]
        else:
            example_input_ids = \
                example['src'] + [self.tokenizer.eos_token_id]
        return torch.tensor(example_input_ids, dtype=torch.long)
    
    
    def _construct_target_ids(self, example):
        '''
        :param example: a dict consisting of 'src' and 'tgt' keys,
            corresponding to source (prompt) and response (target) utterances.
        '''
        if example['tgt']:
            example_target_ids = \
                [self._targets_ignore_index]*(len(example['src'])+1) + example['tgt'] + [self.tokenizer.eos_token_id]
        else:
            example_target_ids = \
                [self._targets_ignore_index]*(len(example['src'])+1)
        return torch.tensor(example_target_ids, dtype=torch.long)
    
    
    def _construct_token_type_ids(self, example):
        '''
        :param example: a dict consisting of 'src' and 'tgt' keys,
            corresponding to source (prompt) and response (target) utterances.
        '''
        if example['tgt']:
            example_token_type_ids = \
                [0] * (len(example['src'])+1) + [1] * (len(example['tgt'])+1)
        else:
            example_token_type_ids = \
                [0] * (len(example['src'])+1)
        return torch.tensor(example_token_type_ids, dtype=torch.long)
    
    
    def _construct_position_ids(self, example):
        '''
        :param example: a dict consisting of 'src' and 'tgt' keys,
            corresponding to source (prompt) and response (target) utterances.
        '''
        if example['tgt']:
            input_len = len(example['src']) + 1 + len(example['tgt'])
            example_position_ids = [i for i in range(input_len)] + [0]
        else:
            input_len = len(example['src']) + 1
            example_position_ids = [i for i in range(input_len)]
        return torch.tensor(example_position_ids, dtype=torch.long)
    
    
    def _construct_attention_mask(self, example):
        '''
        :param example: a dict consisting of 'src' and 'tgt' keys,
            corresponding to source (prompt) and response (target) utterances.
        '''
        if example['tgt']:
            example_attention_mask = \
                [1] * (len(example['src']) + len(example['tgt']) + 1) + [0]
        else:
            example_attention_mask = \
                [1] * (len(example['src']) + 1)
        return torch.tensor(example_attention_mask, dtype=torch.long)
    

    def __call__(self, batch):
        '''
        Duplicates `input_ids` into `labels`, then masks prompt tokens in labels tensor.
        '''

        # join, mask, and generate token type ids
        input_ids       = []
        target_ids      = []
        token_type_ids  = []
        position_ids    = []
        attention_masks = []
        for example in batch:

            # construct input ids: join src and tgt with eos token, and adding one pad token to the right
            example_input_ids_tensor = self._construct_input_ids(example)
            input_ids.append(example_input_ids_tensor)

            # construct target ids: join len(src)+1 `_targets_ignore_index` tokens with tgt and one eos token to the right
            example_target_ids_tensor = self._construct_target_ids(example)
            target_ids.append(example_target_ids_tensor)

            # construct token type ids: 1 indicates response (including trailing eos token), 0 otherwise (prompt and pad)
            example_token_type_ids_tensor = self._construct_token_type_ids(example)
            token_type_ids.append(example_token_type_ids_tensor)

            # construct position ids: simply increment from 0
            example_position_ids_tensor = self._construct_position_ids(example)
            position_ids.append(example_position_ids_tensor)
            
            # construct attention mask: model should not attend to padding
            example_attention_mask_tensor = self._construct_attention_mask(example)
            attention_masks.append(example_attention_mask_tensor)

        # pad
        input_ids       = pad_sequence(input_ids, batch_first=True, padding_value=self._pad_token_id)[:, :self.max_len]
        position_ids    = pad_sequence(position_ids, batch_first=True, padding_value=0)[:, :self.max_len]
        token_type_ids  = pad_sequence(token_type_ids, batch_first=True, padding_value=0)[:, :self.max_len]
        target_ids      = pad_sequence(target_ids, batch_first=True, padding_value=self._pad_token_id)[:, :self.max_len]
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)[:, :self.max_len]
        
        return {
            'input_ids': input_ids, 
            'position_ids': position_ids, 
            'token_type_ids': token_type_ids, 
            'target_ids': target_ids, 
            'attention_masks': attention_masks
        }



class DialogCollateExperimental:
    
    '''
    DIaloGPT training.
    
    collate_fn in dataloader is used for post processing on a single batch, 
    unlike __getitem__ in dataset class which returns a single example.
    '''
    
    def __init__(
        self, 
        tokenizer,
        max_len = 512,
        _targets_ignore_index = -100,
        _pad_token_id = 0
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len  # long sequences will be truncated to this length
        self._targets_ignore_index = _targets_ignore_index
        self._pad_token_id = _pad_token_id
        
    
    def _construct_input_ids(self, example):
        '''
        :param example: a dict consisting of 'src' and 'tgt' keys,
            corresponding to source (prompt) and response (target) utterances.
        '''

        if example['tgt']:
            example_input_ids = \
                example['src'] + [self.tokenizer.eos_token_id] + example['tgt'] + [self._pad_token_id]
        else:
            example_input_ids = \
                example['src'] + [self.tokenizer.eos_token_id]
        return torch.tensor(example_input_ids, dtype=torch.long)
    
    
    def _construct_target_ids(self, example):
        '''
        :param example: a dict consisting of 'src' and 'tgt' keys,
            corresponding to source (prompt) and response (target) utterances.
        '''
        if example['tgt']:
            example_target_ids = \
                example['src'] + [self.tokenizer.eos_token_id] + example['tgt'] + [self.tokenizer.eos_token_id]
        else:
            example_target_ids = \
                example['src'] + [self.tokenizer.eos_token_id]
        return torch.tensor(example_target_ids, dtype=torch.long)
    
    
    def _construct_token_type_ids(self, example):
        '''
        :param example: a dict consisting of 'src' and 'tgt' keys,
            corresponding to source (prompt) and response (target) utterances.
        '''
        if example['tgt']:
            example_token_type_ids = \
                [0] * (len(example['src'])+1) + [1] * (len(example['tgt'])+1)
        else:
            example_token_type_ids = \
                [0] * (len(example['src'])+1)
        return torch.tensor(example_token_type_ids, dtype=torch.long)
    
    
    def _construct_position_ids(self, example):
        '''
        :param example: a dict consisting of 'src' and 'tgt' keys,
            corresponding to source (prompt) and response (target) utterances.
        '''
        if example['tgt']:
            input_len = len(example['src']) + 1 + len(example['tgt'])
            example_position_ids = [i for i in range(input_len)] + [0]
        else:
            input_len = len(example['src']) + 1
            example_position_ids = [i for i in range(input_len)]
        return torch.tensor(example_position_ids, dtype=torch.long)
    
    
    def _construct_attention_mask(self, example):
        '''
        :param example: a dict consisting of 'src' and 'tgt' keys,
            corresponding to source (prompt) and response (target) utterances.
        '''
        if example['tgt']:
            example_attention_mask = \
                [1] * (len(example['src']) + len(example['tgt']) + 1) + [0]
        else:
            example_attention_mask = \
                [1] * (len(example['src']) + 1)
        return torch.tensor(example_attention_mask, dtype=torch.long)
    

    def __call__(self, batch):
        '''
        Duplicates `input_ids` into `labels`, then masks prompt tokens in labels tensor.
        '''

        # join, mask, and generate token type ids
        input_ids       = []
        target_ids      = []
        token_type_ids  = []
        position_ids    = []
        attention_masks = []
        for example in batch:

            # construct input ids: join src and tgt with eos token, and adding one pad token to the right
            example_input_ids_tensor = self._construct_input_ids(example)
            input_ids.append(example_input_ids_tensor)

            # construct target ids: join len(src)+1 `_targets_ignore_index` tokens with tgt and one eos token to the right
            example_target_ids_tensor = self._construct_target_ids(example)
            target_ids.append(example_target_ids_tensor)

            # construct token type ids: 1 indicates response (including trailing eos token), 0 otherwise (prompt and pad)
            example_token_type_ids_tensor = self._construct_token_type_ids(example)
            token_type_ids.append(example_token_type_ids_tensor)

            # construct position ids: simply increment from 0
            example_position_ids_tensor = self._construct_position_ids(example)
            position_ids.append(example_position_ids_tensor)
            
            # construct attention mask: model should not attend to padding
            example_attention_mask_tensor = self._construct_attention_mask(example)
            attention_masks.append(example_attention_mask_tensor)

        # pad
        # input_ids       = pad_sequence(input_ids, batch_first=True, padding_value=self._pad_token_id)[:, :self.max_len]
        input_ids       = pad_sequence(input_ids, batch_first=True, padding_value=0)[:, :self.max_len]
        position_ids    = pad_sequence(position_ids, batch_first=True, padding_value=0)[:, :self.max_len]
        token_type_ids  = pad_sequence(token_type_ids, batch_first=True, padding_value=0)[:, :self.max_len]
        target_ids      = pad_sequence(target_ids, batch_first=True, padding_value=0)[:, :self.max_len]
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)[:, :self.max_len]
        
        return {
            'input_ids': input_ids, 
            'position_ids': position_ids, 
            'token_type_ids': token_type_ids, 
            'target_ids': target_ids, 
            'attention_masks': attention_masks
        }


class CausalLMCollate:
    '''
    Classic GPT2 training.
    '''

    def __init__(
        self, 
        tokenizer,
        max_len = 512
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        '''
        `batch`: a list of dicts, which contain the input strings.
        '''

        batch_tensors = {}
        batch_encoded = self.tokenizer(
            [d['text'] for d in batch],
            truncation = True, 
            max_length = self.max_len, 
            padding = "longest"
        )

        batch_tensors['input_ids'] = torch.tensor(batch_encoded['input_ids'])
        batch_tensors['attention_mask'] = torch.tensor(batch_encoded['attention_mask'])
        batch_tensors['target_ids'] = batch_tensors['input_ids']

        return batch_tensors