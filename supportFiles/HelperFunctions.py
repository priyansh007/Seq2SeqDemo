from nltk.tokenize import WordPunctTokenizer
from torch.utils.data import DataLoader
import pandas as pd
import torch
import supportFiles.Word2Sequence as Word2Sequence

def read_dataset(path="./Dataset/eng-chin.txt"):
    df = pd.read_table(path,header=None).iloc[:,:]
    df = df.drop([2],axis=1)
    df.columns=['english','chinese']

    input_texts = df.english.values.tolist() # This will be all of the English sentences
    target_texts = df.chinese.values.tolist() # This will be all of the Chinese sentences
    return(input_texts, target_texts)

def tokenize_sentences(input_texts, target_texts):
    tk = WordPunctTokenizer()
    #Same can be written as:
    english = [tk.tokenize(sentence.lower()) for sentence in input_texts]
    chinese = [[x for x in sentence] for sentence in target_texts]
    return (english,chinese)
    
    
def build_vocab(english,chinese):    
    input_tokenizer = Word2Sequence.Word2Sequence()
    for words in english:
        input_tokenizer.fit(words)
    input_tokenizer.build_vocab(min=1, max_features=None) #input
    output_tokenizer = Word2Sequence.Word2Sequence()
    for words in chinese:
        output_tokenizer.fit(words)
    output_tokenizer.build_vocab(min=1, max_features=None)
    return(input_tokenizer, output_tokenizer)

def count_max_sentence_len(sentences):
    max_length = 0
    for sentence in sentences:
        if len(sentence) > max_length:
            max_length = len(sentence)
    return max_length

def collate_fn(batch):
    '''
    param batch: ([enc_in, dec_in, dec_out]， [enc_in, dec_in, dec_out], output of getitem...)
    '''
    # unpack values
    enc_in, dec_in, dec_out = list(zip(*batch))
    # Return tensor type
    return torch.LongTensor(enc_in), torch.LongTensor(dec_in), torch.LongTensor(dec_out)

def get_dataloader(dataset, batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn):
    '''
    Returns a way to access and use the data
    '''
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            collate_fn=collate_fn)
    return dataloader
