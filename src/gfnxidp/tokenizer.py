import torch

class Vocab:
    def __init__(self, alphabet) -> None:
        self.stoi = {}
        self.itos = {}
        self.alphabet = alphabet
        for i, alphabet in enumerate(alphabet):
            self.stoi[alphabet] = i
            self.itos[i] = alphabet

class TokenizerWrapper:
    def __init__(self, vocab, args):
        self.args = args
        self.vocab = vocab
        self.vocab_size = len(vocab.alphabet)
    def tokenize(self, seq):
        tokens = []
        for aa in seq:
            tokens.append(self.stoi[aa])
        return tokens
    
    def detokenize(self, tokens):
        unpad_tokens = []
        for token in tokens:
            if token != 20:
                if isinstance(token, int):
                    unpad_tokens.append(token)
                elif isinstance(token, torch.Tensor):
                    unpad_tokens.append(token)
        return ''.join([self.itos[t] for t in unpad_tokens if t in self.itos])
    
    def pad_tokens(self, batch_tokens):
        if isinstance(batch_tokens, tuple):
            batch_tokens = list(batch_tokens)
        lens = [len(batch_tokens[i]) for i in range(len(batch_tokens))]
        max_len = max(lens)
        if max_len != sum(lens) / len(lens):
            max_len = self.args.max_len
            for i in range(len(batch_tokens)):
                if len(batch_tokens[i]) == max_len:
                    pass
                try:
                    batch_tokens[i] = batch_tokens[i] + [len(self.stoi.keys())] * (max_len - len(batch_tokens[i]))
                except Exception as e:
                    print(f"An error occurred: {e}")
                    import pdb; pdb.set_trace();

        return torch.tensor(batch_tokens, dtype=torch.long)

    @property
    def itos(self):
        return self.vocab.itos

    @property
    def stoi(self):
        return self.vocab.stoi


def get_tokenizer(args):
    alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    vocab = Vocab(alphabet)
    tokenizer = TokenizerWrapper(vocab, args)
    return tokenizer