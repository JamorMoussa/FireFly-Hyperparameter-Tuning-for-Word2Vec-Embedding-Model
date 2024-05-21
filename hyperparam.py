from train import train_word_embedding_model
import numpy as np

from firefly import FireFlyOptimizer, FireFlyConfig, FireFlyParameterBounder

import argparse


def fitness(params: np.ndarray) -> float:

    lr = params[0]
    beta1 = params[1]
    beta2 = params[2]
    window_size: int = int(params[3])
    emb_dim: int = int(params[4])

    _, total_loss, _= train_word_embedding_model(
        lr = lr,
        betas= (beta1, beta2),
        window_size= window_size,
        emb_dim= emb_dim,
        epochs= 50
    )

    return total_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a word Embedding Model')

    parser.add_argument('--popsize', type=int, default=5,
                        help="popsize: represent the population size.")
    
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="")
    
    parser.add_argument('--beta0', type=float, default=1.0,
                        help="")

    parser.add_argument('--gamma', type=float, default=0.01,
                        help="")
    
    parser.add_argument('--maxiters', type=int, default=5,
                        help="")
    
    parser.add_argument('--numtest', type=int, default=2,
                        help="")
    
    
    args = parser.parse_args()

    config = FireFlyConfig(
        pop_size= args.popsize,
        alpha= args.alpha,
        beta0= args.beta0,
        gamma= args.gamma,
        max_iters= args.maxiters
    )

    bounder = FireFlyParameterBounder(bounds=[
            (0.0001, 0.01), (0.89, 0.91), (0.98, 0.9999), (2, 10), (2,10)
        ])

    FA = FireFlyOptimizer(config= config, bounder= bounder)
    
    res = []

    for iter  in range(args.numtest):
        FA.run(func= fitness, dim= 5)
        res.append((FA.best_intensity , FA.best_pos))
        print("# test {iter} is finished", "."*30)

    # mean = np.array(res).mean(axis=0)

    best_pos = sorted(res, key= lambda val: val[0])[0][1]

    print(FA.best_intensity)
    print(f"--lr={best_pos[0]} --beta1={best_pos[1]} --beta2={best_pos[2]} --embdim={int(best_pos[3])} --windowsize={int(best_pos[4])}")