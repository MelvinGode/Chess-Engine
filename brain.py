import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import chessgame
import render
import matplotlib.pyplot as plt
import re

# piece_to_id = {piece:i+1 for i,piece in enumerate(["pawn", "rook", "knight", "bishop", "queen", "king"])}
# def board_to_tensor(board):
#     presence_tensor = torch.zeros((8*8), dtype=torch.int)
#     for i,square in enumerate(board.T.ravel()):
#         if square is None : continue
#         presence_tensor[i] = piece_to_id[square.name]
    
#     return presence_tensor.to('cuda')

class Transformer_block(nn.Module):
    def __init__(self, latent_dim:int, num_heads:int):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ff1 = nn.Linear(latent_dim, latent_dim)
        self.activation = nn.GELU()
        self.ff2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, k=None, v = None):
        k = x if k is None else k
        v = x if v is None else v
        x1 = self.norm1(x)
        x1, _ = self.attn(x1, k, v)
        x = x + x1 # residual connection
        x2 = self.norm2(x)
        x2 = self.ff1(x2)
        x2 = self.activation(x2)
        x2 = self.ff2(x2)
        x = x + x2
        return x

class Base_transformer(nn.Module):
    def __init__(self, n_layers:int, latent_dim:int, num_heads:int):
        super(Base_transformer, self).__init__()
        self.positional = nn.Parameter(torch.zeros((1, 64, latent_dim)))
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(13, latent_dim)
        layers = []
        for i in range(n_layers):
            # in_size = latent_dim if i else 8*8*6
            layer = Transformer_block(latent_dim, num_heads)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        # x = board_to_tensor(board)
        x = self.embedding(x) + self.positional
        for layer in self.layers:
            x = layer(x)
        return x
    
class Critic(nn.Module):
    def __init__(self, base: Base_transformer):
        super(Critic, self).__init__()
        self.base = base
        self.classification_layer = nn.Linear(base.latent_dim * 8 * 8, 1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.base(x).view((x.shape[0], -1))
        x = self.classification_layer(x)
        x = self.activation(x)
        return x

class Actor(nn.Module):
    def __init__(self, base: Base_transformer):
        super(Actor, self).__init__()
        self.base = base
        self.decision_layer = nn.Linear(base.latent_dim *8 * 8, 8*8)
        self.logsoftmax = nn.LogSoftmax(1)

    def forward(self, x, legal_positions, iswhite):
        x = self.base(x).view((x.shape[0], -1))
        x = self.decision_layer(x) * (2*iswhite-1)
        mask = torch.full((8*8,), -torch.inf,  device = x.device)
        mask[legal_positions] = 0
        x = x + mask
        x = self.logsoftmax(x)
        return x
    

base = Base_transformer(n_layers=3, latent_dim=128, num_heads=1)

critic = Critic(base)
high_actor = Actor(base)
low_actor = Actor(base)

critic.to('cuda')
high_actor.to('cuda')
low_actor.to('cuda')

def train_self_play(critic, high_actor, low_actor, n_games, gamma = 0.99):
    critic_optimizer = torch.optim.AdamW(params = critic.parameters())
    high_actor_optimizer = torch.optim.AdamW(params = high_actor.parameters())
    low_actor_optimizer = torch.optim.AdamW(params = low_actor.parameters())
    
    critic.train()
    high_actor.train()
    low_actor.train()

    losses = torch.zeros(n_games, 3)
    writer = SummaryWriter()

    for i in range(n_games):
        game = render.Game()
        game_states, piece_log_probs, end_log_probs, winner = game.play_ai_ai_game(low_actor=low_actor, high_actor=high_actor, epsilon=0.15, max_turns=300)
        n_moves = len(game_states)
        estimated_win_probs = critic(game_states.squeeze()).squeeze()
        advantage = torch.cat((estimated_win_probs[1:], torch.tensor([winner]).to("cuda")), 0) - estimated_win_probs.detach()
        advantage = advantage * (2*(torch.arange(n_moves)%2==0) -1).to("cuda")
        # shuffle_index = torch.randperm(n_moves)
        # estimated_win_probs, piece_log_probs, end_log_probs, advantage = estimated_win_probs[shuffle_index].to("cuda"), piece_log_probs[shuffle_index].to("cuda"), end_log_probs[shuffle_index].to("cuda"), advantage[shuffle_index].to('cuda')
        if winner==0: targets = torch.zeros(n_moves, device='cuda')
        else: targets = winner * gamma ** torch.arange(n_moves-1, -1, -1, device="cuda")
        critic_loss = F.mse_loss(estimated_win_probs, targets)

        high_actor_loss = torch.mean(-piece_log_probs * advantage) * 100
        low_actor_loss = torch.mean(-end_log_probs * advantage) * 100
        writer.add_scalar("Critic loss", critic_loss.item(), i)
        writer.add_scalar("High level actor loss", high_actor_loss.item(), i)
        writer.add_scalar("Low level actor loss", low_actor_loss.item(), i)

        critic_optimizer.zero_grad()
        high_actor_optimizer.zero_grad()
        low_actor_optimizer.zero_grad()

        losses[i] = torch.tensor([critic_loss.item(), high_actor_loss.item(), low_actor_loss.item()])
        # print(f'Losses : {losses[i]}')

        critic_loss.backward(retain_graph=True)
        high_actor_loss.backward(retain_graph=True)
        low_actor_loss.backward()

        critic_optimizer.step()
        high_actor_optimizer.step()
        low_actor_optimizer.step()

        print(f"Finished game {i+1}. Outcome: {winner}")

    torch.save(critic, f="models/critic.pt")
    torch.save(high_actor, f="models/high_actor.pt")
    torch.save(low_actor, f="models/low_actor.pt")


def pretrain_critic(critic: Critic, nb_games:int, epochs:int, batch_size:int, test_prop:float, eval_period:int, save_period:int = 0):

    training_data = os.listdir("gamebank/")
    loaded = False
    try:
        maxnbgames = max([int(re.search(r'(?<=_)\d+(?=\.pt)', filename).group(0)) for filename in training_data if re.search(r'(?<=_)\d+(?=\.pt)', filename)])
        state_tensor = torch.load(f"gamebank/state_tensor_{maxnbgames}.pt")
        winner_train = torch.load(f"gamebank/winner_tensor_{maxnbgames}.pt")
        loss_decays = torch.load(f"gamebank/loss_weights_{maxnbgames}.pt")
        if maxnbgames> nb_games:
            state_tensor, winner_train, loss_decays = state_tensor[:nb_games], winner_train[:nb_games], loss_decays[:nb_games]
        startpoint = maxnbgames
        print("Loaded saved training data")
    except Exception as e:
        print(f'Exception {e}')
        startpoint=0
        state_tensor, winner_train, loss_decays = torch.zeros((0, 64), dtype=int), torch.zeros((0,), dtype=int), torch.zeros((0,))

    games, winners = chessgame.extract_pgn_texts("gamebank/lichess_db_standard_rated_2025-09.pgn.zst", nb_games)
    
    state_list, winner_list, loss_list = [state_tensor], [winner_train], [loss_decays]
    for game, winner in zip(games[startpoint:nb_games], winners[startpoint:nb_games]):
        loaded = False
        session = render.Game()
        states = session.load_PGN(game, history_mode=True)
        state_list.append(states)
        winner_list.append(torch.full((len(states),), winner))
        loss_list.append(0.99 ** torch.arange(len(states)-1, -1, -1))

    state_tensor, winner_train, loss_decays = torch.cat(state_list,0), torch.cat(winner_list,0), torch.cat(loss_list,0)
    if not loaded:
        torch.save(state_tensor, f'gamebank/state_tensor_{nb_games}.pt')
        torch.save(winner_train, f'gamebank/winner_tensor_{nb_games}.pt')
        torch.save(loss_decays, f'gamebank/loss_weights_{nb_games}.pt')

    train_cutoff = round(len(state_tensor)*(1-test_prop))
    print("test set size",len(state_tensor)-train_cutoff)
    x_train, x_test = state_tensor[:train_cutoff], state_tensor[train_cutoff:]
    y_train, y_test = winner_train[:train_cutoff], winner_train[train_cutoff:]
    decay_train, decay_test = loss_decays[:train_cutoff], loss_decays[train_cutoff:]
    # print(x_test.shape, y_test.shape, decay_test.shape)
    # print(state_tensor.shape, winner_train.shape, loss_decays.shape)

    optimizer = torch.optim.AdamW(params= critic.parameters())
    critic.train()
    vector_bce = lambda pred, target : target * -torch.log(pred) + (1-target) * torch.log(1-pred)

    train_losses, test_losses = [], []
    step_count = 0
    for epoch in range(epochs):
        print("Epoch",epoch+1)
        perm = torch.randperm(len(x_train))

        for i in tqdm(range(0, len(x_train), batch_size)):
            step_count +=1
            ceiling = min(i+batch_size, len(x_train))

            x_batch = x_train[perm[i:ceiling]].to("cuda")
            y_batch = y_train[perm[i:ceiling]].to("cuda")
            loss_weights = decay_train[perm[i:ceiling]].to("cuda")

            pred = critic(x_batch)
            loss = torch.mean(vector_bce(pred, y_batch) * loss_weights)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step_count % save_period ==0:
                torch.save(critic, f"models/pretrained_critic_step_{step_count}.pt")

            if step_count % eval_period == 0:
                critic.eval()
                pred = critic(x_test.to("cuda"))
                loss = torch.mean(vector_bce(pred, y_test.to("cuda")) * decay_test.to("cuda"))
                test_losses.append(loss.item())
                critic.train()

    torch.save(critic, f"models/pretrained_critic_step_{step_count}.pt")
    try:
        plt.plot(train_losses)
        plt.plot(torch.arange(eval_period, step_count, eval_period), test_losses)
        plt.grid(True)
    except Exception as e:
        print(f'Error for the plot: {e}')

    return critic

# game = render.Game()
# embedlayer = nn.Embedding(7, 128).to("cuda")
# tensorboard = board_to_tensor(game.board.board)
# print(tensorboard)
# embededboard = embedlayer(tensorboard)
# print(embededboard)
# exit()

# random_board_1 = torch.randint(0,6, (64,)).to('cuda')
# random_board_2 = torch.randint(0,6, (64,)).to('cuda')
# print(critic(random_board_1.unsqueeze(0)))
# print(critic(random_board_2.unsqueeze(0)))
# print(random_board_1)
# print(random_board_2)

# exit()

os.chdir(os.path.dirname(os.path.realpath(__file__)))
pretrain_critic(critic, nb_games=10_000, epochs=10, batch_size=100, test_prop=0.01, eval_period=100, save_period=10000)
# train_self_play(critic, high_actor, low_actor, 1_000)

# TODO
# Update after batch of games with random selection
# Add dropout