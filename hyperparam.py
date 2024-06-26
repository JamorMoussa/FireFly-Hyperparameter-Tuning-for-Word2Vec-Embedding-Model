from train import train_word_embedding_model
import numpy as np

from firefly import FireFlyOptimizer, FireFlyConfig, FireFlyParameterBounder
import generateReport as gr

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
        epochs= 10
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
            (0.00001, 0.01), (0.899, 0.901), (0.9989, 0.999), (2, 10), (1,3)
        ])

    FA = FireFlyOptimizer(config= config, bounder= bounder)
    
    results = FA.run(func= fitness, dim= 5)

    best_pos = FA.best_pos

    print("best intensity:", FA.best_intensity)

    gr.plot_loss(dir=gr.REPORT_DIR, results= results)

    print(f"--lr={best_pos[0]} --beta1={best_pos[1]} --beta2={best_pos[2]} --embdim={int(best_pos[3])} --windowsize={int(best_pos[4])}")