import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter

import chessgame
import render

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
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.base(x).view((x.shape[0], -1))
        x = self.classification_layer(x)
        x = self.tanh(x)
        return x

class Actor(nn.Module):
    def __init__(self, base: Base_transformer):
        super(Actor, self).__init__()
        self.base = base
        self.decision_layer = nn.Linear(base.latent_dim *8 * 8, 8*8)
        self.softmax = nn.Softmax(1)

    def forward(self, x, legal_positions, iswhite):
        x = self.base(x).view((x.shape[0], -1))
        x = self.decision_layer(x) * (2*iswhite-1)
        mask = torch.full(8*8, -torch.inf,  device = x.device)
        mask[legal_positions] = 0
        x = x + mask
        x = self.softmax(x)
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
        if winner==0: targets = torch.zeros(n_moves, devide='cuda')
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
train_self_play(critic, high_actor, low_actor, 1_000)

# TODO
# Update after batch of games with random selection
# Add dropout
# Add skip connections