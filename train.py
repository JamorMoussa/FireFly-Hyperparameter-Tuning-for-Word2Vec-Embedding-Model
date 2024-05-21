import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data import sentences, get_context_target_vocab

import argparse

from tqdm import tqdm


class SumReshapeTransform(nn.Module):

    def __init__(self,) -> None:
        super(SumReshapeTransform, self).__init__()

    def forward(self, input: torch.Tensor):
        return input.sum(dim=1).reshape(1, -1)


def train_word_embedding_model(
        lr: float,
        emb_dim: int, 
        betas: tuple[float, float],
        window_size: int, 
        epochs: int = 10
    ):

    context, target, _, vocab_size = get_context_target_vocab(sentences=sentences, w=window_size)

    dataset = TensorDataset(context, target)

    data_loader = DataLoader(dataset, batch_size=1)

    model = nn.Sequential(
        nn.Embedding(vocab_size, emb_dim),
        SumReshapeTransform(),
        nn.Linear(emb_dim, vocab_size),
    )

    criterion = nn.CrossEntropyLoss()

    opt = optim.Adam(model.parameters(), lr=lr, betas=betas)

    for _ in (bar := tqdm(range(epochs))):
        
        total_loss = 0

        for context, target in data_loader:

            opt.zero_grad()
            
            pred = model(context)

            target_vec = torch.zeros(1, vocab_size)

            target_vec[0][int(target)] = 1
            
            loss = criterion(pred, target_vec)

            loss.backward()
                
            opt.step()

            total_loss += loss.item()
        
        bar.set_description(f"Loss: {total_loss :.5f}")

    return model, total_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a word Embedding Model')

    parser.add_argument('--lr', type=float, default=0.01,
                        help="lr: represent the learning rate.")
    
    parser.add_argument('--beta1', type=float, default=0.9,
                        help="")
    
    parser.add_argument('--beta2', type=float, default=0.99,
                        help="")
    
    parser.add_argument('--windowsize', type=int, default=1,
                        help="")
    
    parser.add_argument('--embdim', type=int, default=2,
                        help="")
    
    parser.add_argument('--epochs', type=int, default=1,
                        help="epochs for the training loop")
    
    args = parser.parse_args()


    model, total_loss = train_word_embedding_model(
        lr= args.lr,
        betas= (args.beta1, args.beta2),
        window_size= args.windowsize,
        emb_dim= args.embdim,
        epochs= args.epochs
    )

    print(model)
    print(total_loss)
    
    