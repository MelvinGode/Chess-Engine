import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import chessgame
import render

piece_to_id = {piece:i+1 for i,piece in enumerate(["pawn", "rook", "knight", "bishop", "queen", "king"])}
def board_to_tensor(board):
    presence_tensor = torch.zeros((8*8), dtype=torch.int)
    for i,square in enumerate(board.T.ravel()):
        if square is None : continue
        presence_tensor[i] = piece_to_id[square.name]
    
    return presence_tensor.to('cuda')

class Transformer_block(nn.Module):
    def __init__(self, latent_dim:int, num_heads:int):
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, num_heads)
        self.ff1 = nn.Linear(latent_dim, latent_dim)
        self.activation = nn.GELU()
        self.ff2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, k=None, v = None):
        k, v= x if k is None else k, x if v is None else v
        x, _ = self.attn(x, k, v)
        x = self.ff1(x)
        x = self.activation(x)
        x = self.ff2(x)
        return x

class Base_transformer(nn.Module):
    def __init__(self, n_layers:int, latent_dim:int, num_heads:int):
        super(Base_transformer, self).__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(7, latent_dim)
        layers = []
        for i in range(n_layers):
            # in_size = latent_dim if i else 8*8*6
            layer = Transformer_block(latent_dim, num_heads)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
    
    def forward(self, board):
        x = board_to_tensor(board)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
class Critic(nn.Module):
    def __init__(self, base: Base_transformer):
        super(Critic, self).__init__()
        self.base = base
        self.classification_layer = nn.Linear(base.latent_dim * 8 * 8, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.base(x).ravel()
        x = self.classification_layer(x)
        x = self.sigmoid(x)
        return x

class Actor(nn.Module):
    def __init__(self, base: Base_transformer):
        super(Actor, self).__init__()
        self.base = base
        self.decision_layer = nn.Linear(base.latent_dim *8 * 8, 8*8)
        self.softmax = nn.Softmax(0)

    def forward(self, x, legal_positions, iswhite):
        x = self.base(x).ravel()
        mask = torch.zeros(8*8, device = x.device)
        mask[legal_positions] = 1
        x = self.decision_layer(x) * (2*iswhite-1)
        x = self.softmax(x)
        x = x * mask
        return x
    

base = Base_transformer(n_layers=3, latent_dim=128, num_heads=1)

critic = Critic(base)
high_actor = Actor(base)
low_actor = Actor(base)

critic.to('cuda')
high_actor.to('cuda')
low_actor.to('cuda')

def train_self_play(critic, high_actor, low_actor, n_games):
    critic_optimizer = torch.optim.AdamW(params = critic.parameters())
    high_actor_optimizer = torch.optim.AdamW(params = high_actor.parameters())
    low_actor_optimizer = torch.optim.AdamW(params = low_actor.parameters())
    
    critic.train()
    high_actor.train()
    low_actor.train()

    losses = torch.zeros(n_games, 3)

    for i in range(n_games):
        game = render.Game()
        estimated_win_probs, piece_log_probs, end_log_probs, winner = game.play_ai_ai_game(critic = critic, low_actor=low_actor, high_actor=high_actor, epsilon=0.15)
        n_moves = len(estimated_win_probs)
        estimated_win_probs = estimated_win_probs.squeeze()
        advantage = torch.cat((estimated_win_probs[1:], torch.tensor([winner]).to("cuda")), 0) - estimated_win_probs
        advantage = advantage * (2*(torch.arange(n_moves)%2) -1).to("cuda")
        # shuffle_index = torch.randperm(n_moves)
        # estimated_win_probs, piece_log_probs, end_log_probs, advantage = estimated_win_probs[shuffle_index].to("cuda"), piece_log_probs[shuffle_index].to("cuda"), end_log_probs[shuffle_index].to("cuda"), advantage[shuffle_index].to('cuda')
        critic_loss = F.binary_cross_entropy(estimated_win_probs, torch.tensor([winner], dtype=torch.float).repeat(n_moves).to('cuda'))
        high_actor_loss = torch.mean(-piece_log_probs * advantage)
        low_actor_loss = torch.mean(-end_log_probs * advantage)

        critic_optimizer.zero_grad()
        high_actor_optimizer.zero_grad()
        low_actor_optimizer.zero_grad()

        losses[i] = torch.tensor([critic_loss.item(), high_actor_loss.item(), low_actor_loss.item()])

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


# game = render.Game()
# embedlayer = nn.Embedding(7, 128).to("cuda")
# tensorboard = board_to_tensor(game.board.board)
# print(tensorboard)
# embededboard = embedlayer(tensorboard)
# print(embededboard)
# exit()

os.chdir(os.path.dirname(os.path.realpath(__file__)))
train_self_play(critic, high_actor, low_actor, 1_000)

# TODO
# Can save board states and compute win probabilities after game
# Add batch/layer norm
# Add dropout
# Add skip connections