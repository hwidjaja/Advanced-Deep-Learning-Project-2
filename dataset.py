import torch
from torch.utils.data import Dataset
import re 
from nltk.tokenize import TweetTokenizer


class DialogLMDataset(Dataset):

    def __init__(
        self,
        df,
        src_col,
        src_eos_token,
        tokenizer,
        tgt_col = None,
    ):

        self.tokenizer = tokenizer
        self.train_mode = False
        self.tgts = None

        if tgt_col is not None:
            self.train_mode = True

        len_df = len(df)
        srcs = []
        tgts = []
        num_proc = 0
        for i, row in df.iterrows():

            if num_proc % 1000 == 0:
                print(f'\rProcessing row: {num_proc} of {len_df}', end='', flush=True)

            num_proc += 1

            src = DialogLMDataset.preprocess_text_minimal(
                txt = str(row[src_col]),
                dataset_eos_token = src_eos_token,
                tokenizer_eos_token = self.tokenizer.eos_token
            )
            src_enc = tokenizer.encode(src)
            srcs.append(src_enc)

            # preprocess and tokenize tgt col if available (usually for training)
            if tgt_col is not None:
                tgt = DialogLMDataset.preprocess_text_minimal(row[tgt_col])
                tgt_enc = tokenizer.encode(tgt)
                tgts.append(tgt_enc)

        # save
        self.srcs = srcs
        self.tgts = tgts


    @staticmethod
    def preprocess_text_minimal(
        txt,
        dataset_eos_token = 'EOS',
        tokenizer_eos_token = '<|endoftext|>'
    ):
        # remove "title : " prefixes
        if txt[:8] == 'title : ':
            txt = txt[8:]
        
        # replace 'EOS' with tokenizer's EOS
        if dataset_eos_token != tokenizer_eos_token:
            txt_utterance_split = txt.split(dataset_eos_token)
            txt = tokenizer_eos_token.join([s.strip() for s in txt_utterance_split])
        
        return txt


    @staticmethod
    def preprocess_text(
        txt,
        dataset_eos_token = 'EOS',
        tokenizer_eos_token = '<|endoftext|>'
    ):
        # remove "title : " prefixes
        if txt[:8] == 'title : ':
            txt = txt[8:]
        
        txt = str(txt).lower()
        
        # url and tag
        words = []
        for word in txt.split():
            if word[0] == '#': # don't allow tag
                continue
            i = word.lower().find('http')
            if i >= 0:
                word = word[:i] + ' ' + '__url__'
            words.append(word.strip())
        txt = ' '.join(words)

        # remove illegal char
        txt = txt.replace(chr(92),'') # chr(92) = '\'. as twitter has 'b\/c' rather than 'b/c'
        txt = txt.replace("b/c","because").replace('j/k','just kidding').replace('w/o','without').replace('w/','with')
        txt = re.sub('__mention__','MENTION',txt)
        txt = re.sub('__url__','URL',txt)
        txt = re.sub(r"[^A-Za-z0-9()\[\]:,.!?'“” ]", " ", txt)
        txt = re.sub('MENTION','__mention__',txt)
        txt = re.sub('URL','__url__',txt)

        tokenizer = TweetTokenizer(preserve_case=True)
        txt = ' ' + ' '.join(tokenizer.tokenize(txt)) + ' '
        
        # remove un-necessary space
        txt = ' '.join(txt.split())
        
        # replace 'EOS' with tokenizer's EOS
        if dataset_eos_token != tokenizer_eos_token:
            txt_utterance_split = txt.split(dataset_eos_token.lower())
            txt = tokenizer_eos_token.join([s.strip() for s in txt_utterance_split])
        
        return txt
        
        
    def __len__(self):
        return len(self.srcs)
    
    
    def __getitem__(self, index):
        '''
        __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalized source and
        target values using the vocabulary objects we created in __init__
        '''

        if self.train_mode:
            return {'src': self.srcs[index], 'tgt': self.tgts[index]}
        else:
            return {'src': self.srcs[index], 'tgt': None}



class CausalLMDataset(Dataset):

    def __init__(
        self,
        df,
        text_col,
        bos_token = '<|startoftext|>',
        eos_token = '<|endoftext|>'
    ):

        """
        A simple dataset for self-supervised Causal Language Modeling.
        """

        self.train_mode = True
        self.eos_token  = eos_token
        self.bos_token  = bos_token

        len_df = len(df)
        texts = []
        num_proc = 0
        for i, row in df.iterrows():

            if num_proc % 1000 == 0:
                print(f'\rProcessing row: {num_proc} of {len_df}', end='', flush=True)

            num_proc += 1

            text = CausalLMDataset.preprocess_text(row[text_col])
            text = self.bos_token + text + self.eos_token
            texts.append(text)

        # save
        self.texts = texts


    @staticmethod
    def preprocess_text(txt):
        txt = str(txt).lower()

        # url and tag
        words = []
        for word in txt.split():
            if word[0] == '#': # don't allow tag
                continue
            i = word.lower().find('http')
            if i >= 0:
                word = word[:i] + ' ' + '__url__'
            words.append(word.strip())
        txt = ' '.join(words)

        # remove illegal char
        txt = txt.replace(chr(92),'') # chr(92) = '\'. as twitter has 'b\/c' rather than 'b/c'
        txt = txt.replace("b/c","because").replace('j/k','just kidding').replace('w/o','without').replace('w/','with')
        txt = re.sub('__mention__','MENTION',txt)
        txt = re.sub('__url__','URL',txt)
        txt = re.sub(r"[^A-Za-z0-9()\[\]:,.!?'“” ]", " ", txt)
        txt = re.sub('MENTION','__mention__',txt)
        txt = re.sub('URL','__url__',txt)

        tokenizer = TweetTokenizer(preserve_case=True)
        txt = ' ' + ' '.join(tokenizer.tokenize(txt)) + ' '

        # remove un-necessary space
        return ' '.join(txt.split())
        
        
    def __len__(self):
        return len(self.texts)
    
    
    def __getitem__(self, index):
        '''
        __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalized source and
        target values using the vocabulary objects we created in __init__
        '''

        return {'text': self.texts[index]}