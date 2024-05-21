import torch

sentences = [
    "The king is a man.",
    "The queen is a woman.",
    "The queen is the wife of the king.",
    "A man can be a king.",
    "A woman can be a queen.",
    "A king is not a queen, and a queen is not a king.",
    "The prince is the son of the king and queen.",
    "The princess is the daughter of the king and queen.",
    "A prince can become a king.",
    "A princess can become a queen.",
    "The castle is the home of the king and queen.",
    "The kingdom is ruled by the king.",
    "The throne is where the king sits.",
    "The crown is worn by the king and queen.",
    "The knight serves the king.",
    "The kingdom has many subjects.",
    "The king leads his army.",
    "The queen attends royal events.",
    "The prince trains to become a knight.",
    "The princess learns to rule the kingdom.",
    "The king and queen host a grand ball.",
    "The royal family lives in the palace.",
    "The subjects are loyal to the king and queen.",
    "The kingdom is prosperous under the king's rule.",
    "The king commands respect from everyone.",
    "The queen is known for her wisdom.",
    "The prince dreams of adventure.",
    "The princess is admired for her beauty.",
    "The king decrees a new law.",
    "The queen advises the king on important matters."
]

def get_context_target_vocab(
        sentences: list[str],
        w: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:

    context = []
    target = []

    text = " ".join(sentences).lower()
    
    vocab = {word: i for i, word in enumerate(set(text.split()))}

    words = text.split()
    for i in range(1, len(words) - w):
        target.append(vocab[words[i]])
        context.append([vocab[words[i-w]], vocab[words[i+w]]])

    return torch.tensor(context), torch.tensor(target), vocab, len(vocab)