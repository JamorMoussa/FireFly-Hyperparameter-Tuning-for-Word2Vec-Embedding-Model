import torch


sentences = [
    "  king is a man ",
    "  queen is a woman ",
    "  queen is  wife of  king ",
    "  man can be a king ",
    "  woman can be a queen ",
    "  king is not a queen and a queen is not a king ",
    "  prince is  son of  king and queen ",
    "  princess is  daughter of king and queen ",
    "  prince can become a king ",
    "  prince is the son of king and queen ",
    "  prince is brother of princess ",
    "  princess is sister of prince ",
    "  prince is the daughter of king and queen ",
    "  princess can become a queen ",
    "  castle is home of king and queen ",
    "  kingdom is ruled by king ",
]





def get_context_target_vocab(
        sentences: list[str],
        w: int = 1
    ) -> tuple[torch.tensor, torch.tensor, dict]:

    context = []
    target = []

    text = " ".join(sentences).lower()
    
    vocab = {word: i for i, word in enumerate(set(text.split()))}

    words = text.split()
    for i in range(1, len(words) - w):
        target.append(vocab[words[i]])
        context.append([vocab[words[i-w]], vocab[words[i+w]]])

    return torch.tensor(context), torch.tensor(target), vocab, len(vocab)