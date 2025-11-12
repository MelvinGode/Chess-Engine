import pygame
import os
import chessgame
import numpy as np
from copy import deepcopy
from pygame import mixer
import time
import torch

capital_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
letters = [letter.lower() for letter in capital_letters]
letter_indexes = {letters[i]:i for i in range(len(letters))}
piece_full_names = {letter:full for full, letter in zip(["pawn", "king", "knight", "rook", "bishop", "queen"], ["", "K", "N", "R", "B", "Q"])}

pygame.init()

WIDTH, HEIGHT = 1000, 800
corner_x = 50
corner_y = 50
margin = np.array([corner_x, corner_y])
square_size = 70

reset_button_coords = margin + np.array([square_size * 8, 30]) + np.array([100, 0])
reset_button_size = np.array([170, 60])


class Game:
    def __init__(self):
        self.board = chessgame.create_classic_board()
        self.nb_pieces = np.sum(self.board.board != None)
        self.check = [False, False]
        self.gameover = False
        self.PGN = ' '
        self.start_time = time.time()
        self.turn_counter = 0
        self.playing_color = 1
        self.current_PGN_index = 0

        os.chdir(os.path.dirname(os.path.realpath(__file__)))

    def load_PGN(self, PGN_string:str):
        self.PGN = PGN_string
        self.board = chessgame.create_classic_board()
        playing_side = 0
        current_turn = 0
        for item in PGN_string.split(" "):
            if not item : break
            dotloc = item.find(".")
            if dotloc>0:
                current_turn = int(item[:dotloc])
                playing_side = 1-item.endswith('...')
                item = item[dotloc + 1 + 2*(1-playing_side):]
            else:
                playing_side = 1-playing_side

            if item.startswith('O-O'): # Castling
                if item == 'O-O-O': start, end = [4, playing_side*7], [2, playing_side*7]
                elif item == "O-O": start, end = [4, playing_side*7], [6, playing_side*7]
            else:
                item = item.replace("x","").replace("+","").replace("#","")
                end = [letter_indexes[item[-2]], 8-int(item[-1])]
                ispawn = item[0] != item[0].capitalize()
                piece_letter = item[0] if not ispawn else ""
                fakepiece = chessgame.Piece(end[0], end[1], name = piece_full_names[piece_letter], color=playing_side if piece_letter else 1-playing_side) # If pawn, set to black for it to be moving in the backwards direction
                fakepiece.hasmoved = True
                possible_sources = self.board.get_piece_legal_moves(fakepiece, collide_mode=True)
                if not len(possible_sources): raise ValueError(f"Error: invalid move at {'white' if playing_side else 'black'}\'s move, turn {current_turn}. Instruction \'{item}\' ")
                elif len(possible_sources)==1: start = possible_sources[0]
                else: # ambiguity
                    if len(item) == 4 - ispawn:
                        try:
                            startrank = int(item[1 - ispawn])
                            start = possible_sources[possible_sources[:,1]==startrank][0]
                        except Exception:
                            startfile = letter_indexes[item[1 - ispawn]]
                            start = possible_sources[possible_sources[:,0]==startfile][0]
                    elif len(item) == 5 :
                        start = [letter_indexes[item[1 - ispawn]], int(item[2 - ispawn])]

            illegal = self.board.move(start, end)
            if illegal: raise ValueError(f"Error: invalid move at {'white' if playing_side else 'black'}\'s move, turn {current_turn}. Instruction \'{item}\' ")
        self.turn_counter = int(current_turn)
        print(f"Loaded PGN, turn {self.turn_counter}")
        self.current_PGN_index = len(PGN_string)
        self.playing_color = 1-playing_side


    def _shift_game_state(self, forward:bool):
        PGN_temp = self.PGN
        if forward: 
            space_index = self.PGN[self.current_PGN_index:].find(' ')
        else : space_index = self.PGN[:max(self.current_PGN_index-1, 0)].rfind(' ')
        if space_index == -1:
            if forward : new_PGN_index = len(self.PGN)
            else : new_PGN_index = 0
            if self.current_PGN_index == new_PGN_index : return
            self.current_PGN_index = new_PGN_index
        else:
            self.current_PGN_index = (not forward)* space_index + forward* (self.current_PGN_index + space_index) +1
        if not forward : self.gameover = False

        self.load_PGN(self.PGN[:self.current_PGN_index])
        self.PGN = PGN_temp


    def _create_window(self):
        #window
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Chess Engine')
        #text
        pygame.font.get_init()
        self.font = pygame.font.SysFont('freesanbold.ttf', 50)
        self.smallfont = pygame.font.SysFont('freesan.ttf', 30)
        whoseturnisit = self.font.render(f"{'White' if self.playing_color else 'Black'}\'s move", True, (255,255,255))
        self.turn_rect = whoseturnisit.get_rect()
        self.turn_rect.center = ((self.board.width * square_size)//2 + corner_x, self.board.height* square_size + corner_y + 100)
        winnertext = self.font.render(f"Checkmate, {'White' if 1-self.playing_color else 'Black'} wins", True, (255,255,255))
        self.winner_rect = winnertext.get_rect()
        self.winner_rect.center = ((self.board.width * square_size)//2 + corner_x, self.board.height* square_size + corner_y + 100)
        #button
        self.reset_button = Button(reset_button_coords[0], reset_button_coords[1], reset_button_size[0], reset_button_size[1], game=self, function=self._reset_game, text="New Game", color=(150, 150, 150), margin = .1)
        self.rewind_button = Button(corner_x + 8*square_size + 100, corner_y + 100, 50, 50, game=self, function=self._shift_game_state, args=(False,), text="P", color=(150, 150, 150), margin = .1)
        self.forward_button = Button(corner_x + 8*square_size + 220, corner_y + 100, 50, 50, game=self, function=self._shift_game_state, args=(True,), text="F", color=(150, 150, 150), margin = .1)
        self.buttons = [self.reset_button, self.rewind_button, self.forward_button]
        #images
        self.piece_images = {}
        for img_name in os.listdir("Assets/Pieces/"):
            self.piece_images[img_name[:-4]] = pygame.transform.scale_by(pygame.image.load(f"Assets/Pieces/{img_name}"), 1/2)
        #sound
        mixer.init()
        mixer.Channel(1).set_volume(0.5)

    def _draw_grid(self):
        for i in range(0, 64, 1):
            y = (i//8) * square_size +corner_y
            x = (i%8) * square_size +corner_x
            isgrey = (i%2)
            if (i//8)%2 : isgrey = 1-isgrey
            color =  (50,50,50) if isgrey else (255, 255, 255)
            pygame.draw.rect(self.window, color, [x, y, square_size, square_size], 0)
        pygame.draw.rect(self.window, (255, 255, 255), [corner_x, corner_y, square_size*8, square_size*8], 1)

    def _dispay_coords(self):
        texts = []
        coords = []
        for i, letter in zip(range(8), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']):
            texts.append(self.font.render(letter, True, (255, 255, 255)))
            coords.append([corner_x + i*square_size + 35,  corner_y + square_size*8 + 35])

            texts.append(self.font.render(str(i+1), True, (255, 255, 255)))
            coords.append([corner_x //2, corner_y + (7-i)*square_size + 35])

        for text, coord in zip(texts, coords):
            textrect = text.get_rect()
            textrect.center = (coord[0], coord[1])
            self.window.blit(text, textrect)

    def _display_pieces(self):
        for piece in self.board.board.ravel():
            if piece is None: continue
            color = "white" if piece.color else 'black'
            sprite = self.piece_images[color+'-'+piece.name]
            coords = np.array([piece.x, piece.y])*square_size + square_size//2 + margin
            rect = sprite.get_rect()
            rect.center = (coords[0], coords[1])
            self.window.blit(sprite, rect)

    def _display_board(self): # TODO change to board.show(), need to modify board class and pass the right arguments initialized in create_window()
        self.window.fill(0)
        self._draw_grid()
        self._display_pieces()
        self._dispay_coords()
        [button.show() for button in self.buttons]
        whoseturnisit = self.font.render(f"{'White' if self.playing_color else 'Black'}\'s move", True, (255,255,255))
        if not self.gameover: self.window.blit(whoseturnisit, self.turn_rect)
        else : 
            winnertext = self.font.render(f"Checkmate, {'White' if 1-self.playing_color else 'Black'} wins", True, (255,255,255))
            self.window.blit(winnertext, self.winner_rect)

        pygame.display.update()

    def _save_PGN(self, filepath:str=''):
        if filepath and not (filepath.endswith('/') or filepath.endswith('\\')): filepath = filepath+'/'
        with open(f"PGN/{filepath}{self.start_time}.pgn", "w", encoding="utf-8") as f:
            f.write(self.PGN)
    
    def _reset_game(self):
        self.board = chessgame.create_classic_board()
        self.nb_pieces = np.sum(self.board.board != None)
        self.start_time = time.time()
        self.PGN = ' '
        self.current_PGN_index = 0
        self.playing_color = 1
        self.turn_counter = 0
        self.selected_piece = None
        mixer.Channel(1).play(pygame.mixer.Sound("Assets/Sounds/reset.mp3"))

    def _update_PGN(self, start, end, filepath:str=''):
        self.PGN = self.PGN[:self.current_PGN_index]
        castled = False
        if 1 - self.playing_color:
            self.turn_counter +=1
            self.PGN += f'{self.turn_counter}.'
        #ambiguity check
        if self.selected_piece.name=="king" and abs(end[0] - start[0])>1.5 : # Get castling out of the way as it is the only scenario where a move is not reversible
            if end[0] > start[0]:self.PGN += 'O-O '
            else : self.PGN +='O-O-O '
            castled = True
        else:
            self.PGN += self.selected_piece.PGN_letter
            ambiguity = False
            file_ambiguity, rank_ambiguity = False, False
            for piece in self.previous_board.board.ravel():
                if piece is None or piece.name!=self.selected_piece.name or piece.color!=self.selected_piece.color or (piece.x==start[0] and piece.y==start[1]): continue
                if np.any( np.all(self.previous_board.get_piece_legal_moves(piece)[:,1,:] == np.array(end), axis=1) ):
                    ambiguity = True
                    if piece.x == self.selected_piece.x: file_ambiguity = True
                    elif piece.y == self.selected_piece.y: rank_ambiguity = True
            if ambiguity:
                if not file_ambiguity:
                    self.PGN += letters[start[0]]
                elif not rank_ambiguity:
                    self.PGN += str(8-start[1])
                else:
                    self.PGN += f"{letters[start[0]]}{8-start[1]}"
        if np.sum(self.board.board != None) < self.nb_pieces: self.PGN += 'x'# If there has been a capture
        if not castled: self.PGN += f"{letters[end[0]]}{8-end[1]} "
        if self.gameover and not self.PGN.endswith('#') : self.PGN[-1] = '#'
        self._save_PGN(filepath)
        self.current_PGN_index = len(self.PGN)
    
    def play_human_human_game(self):
        self._create_window()

        running=True
        self.selected_piece = None

        self._display_board()

        self.gameover = False
        while running:
            for event in pygame.event.get():
                if self.gameover:
                    self.PGN = self.PGN[:-1] + '#' + f'{0 if self.playing_color else 1}-{1 if self.playing_color else 0}'
                    self._display_board()

                    if event.type == pygame.QUIT:
                        running=False
                        break

                    if event.type==pygame.MOUSEBUTTONDOWN:
                        for button in self.buttons: button.trigger_if_pressed(pygame.mouse.get_pos())

                    continue

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_coords = np.array(pygame.mouse.get_pos())
                    for button in self.buttons: button.trigger_if_pressed(mouse_coords)

                    board_coords = (mouse_coords - margin) // square_size

                    # If we are already holding a piece in hand
                    if self.selected_piece is not None:
                        try: 
                            previous_coords = [self.selected_piece.x, self.selected_piece.y]
                            self.previous_board = deepcopy(self.board)
                            illegal_flag = self.board.move([self.selected_piece.x, self.selected_piece.y], board_coords)
                            if illegal_flag:
                                self.selected_piece = None
                                if illegal_flag==2: print("Suicide move, illegal")
                                continue
                            else : self.playing_color = 1-self.playing_color
                        except Exception as e:
                            print(f"Error : {e}")
                            self.selected_piece = None
                            continue
                        finally:
                            self._display_board()
                        
                        print(f'{str(self.selected_piece).replace("at", "to")} from {letters[previous_coords[0]]}{8-previous_coords[1]}')

                        #mate checker
                        next_moves = self.board.get_all_legal_moves(self.playing_color)
                        self.gameover = True
                        for move in next_moves: # Check if next player has at least one legal move
                            fake_board = deepcopy(self.board)
                            illegality_flag = fake_board.move(move[0], move[1])
                            if not illegality_flag:
                                self.gameover = False
                                break

                        #PGN update
                        self._update_PGN(previous_coords, board_coords)
                        if np.sum(self.board.board != None) < self.nb_pieces: # If there has been a capture
                            self.nb_pieces -= 1
                            mixer.Channel(0).play(pygame.mixer.Sound("Assets/Sounds/capture.mp3"))
                        else: mixer.Channel(0).play(pygame.mixer.Sound("Assets/Sounds/move.mp3"))

                        #Turn change
                        self.selected_piece = None                        

                        #check checker
                        # if check[playing_color]: 
                        #     check[playing_color] = False # You cannot have stayed in check on your opponent's turn
                        #     continue
                        # opponent_moves = board.get_all_legal_moves(1-playing_color)
                        # for landing in next_moves[:,1,:]:
                        #     if board.board[landing[0], landing[1]].name=="king" and board.board[landing[0], landing[1]].color != playing_color: #Check for color needed because castling can land a piece on where king currently is
                        #         white_check = True
                    
                    else:# If we just picked up a piece
                        self._display_board()
                        #convert x y coords in board coords
                        try: self.selected_piece = self.board.board[board_coords[0], board_coords[1]]
                        except Exception: continue
                        if self.selected_piece is None or self.selected_piece.color != self.playing_color: 
                            self.selected_piece = None
                            continue
                        # print(f"Selected {'white' if self.selected_piece.color else 'black'} {self.selected_piece.name}")
                        moves = self.board.get_piece_legal_moves(self.selected_piece)
                        end_coords = moves[:, 1, :]
                        if len(end_coords)==0:
                            self.selected_piece = None
                        # color legal squares
                        for board_coords in end_coords:
                            xy_coords = board_coords * square_size + margin
                            highlight_color = (255, 100, 100)
                            pygame.draw.rect(self.window, highlight_color, [xy_coords[0], xy_coords[1], square_size, square_size], 2)
                            # pygame.draw.rect(window, (255, 255, 255), [xy_coords[0], xy_coords[1], square_size, square_size], 3)

                if event.type == pygame.QUIT:
                    running=False

                pygame.display.update()

    def play_ai_ai_game(self, high_actor, low_actor, epsilon:float, max_turns = 300):
        self.gameover = False
        game_states = []
        piece_log_probs = []
        end_log_probs = []

        turn_count = 0
        while not self.gameover and turn_count<max_turns:
            #mate checker
            next_moves = self.board.get_all_legal_moves(self.playing_color)
            legal_moves = []
            self.gameover = True
            for move in next_moves: # Check if next player has at least one legal move
                fake_board = deepcopy(self.board)
                illegality_flag = fake_board.move(move[0], move[1])
                if not illegality_flag:
                    self.gameover = False
                    legal_moves.append(move)
            if self.gameover: break
            legal_moves = np.array(legal_moves, dtype=int)
            legal_moves = 8* legal_moves[:,:, 0] + legal_moves[:,:,1] #convert to 1D

            current_state = self.board.tensorboard.clone().to('cuda').unsqueeze(0)

            game_states.append(current_state)
            
            selected_start_probas = high_actor(current_state, np.unique(legal_moves[:,0]), self.playing_color).squeeze()
            explore =  np.random.random()< epsilon
            if not explore: selected_start = selected_start_probas.argmax().to("cpu")
            else: selected_start = np.random.choice(np.unique(legal_moves[:,0]))
            try:
                selected_end_probas = low_actor(current_state, legal_moves[np.isclose(legal_moves[:,0], selected_start), 1], self.playing_color).squeeze()
            except Exception as e:
                print(f'legal moves: {legal_moves}')
                print(f'Selected start: {selected_start}')
                print(f'Explored: {explore}')
                print(f"Error {e}")
                raise ValueError("ahh")
            if np.random.random()>= epsilon: 
                selected_end = selected_end_probas.argmax().to("cpu")
            else: selected_end = np.random.choice(np.unique(legal_moves[np.isclose(legal_moves[:,0], selected_start), 1]))

            piece_log_probs.append(torch.log(selected_start_probas[selected_start]))
            end_log_probs.append(torch.log(selected_end_probas[selected_end]))

            start = np.array([selected_start//8, selected_start%8])
            end = np.array([selected_end//8, selected_end%8])
            self.previous_board = deepcopy(self.board)
            self.selected_piece = self.board.board[start[0], start[1]]
            # print(f'start : {start}, end : {end}')
            if self.board.move(start, end):
                print(f'Legal moves : {legal_moves}')
                print(f'Selected start: {selected_start}. Explored: {explore}')
                print(f'selected start: {selected_start}, 1st Mask: {np.unique(legal_moves[:,0])}. Selected end: {selected_end}. 2nd Mask: {legal_moves[legal_moves[:,0]==selected_start, 1]}')
                raise ValueError(f"AI somehow played an illegal move: {start} to {end}. ")

            #Turn change
            self.playing_color = 1-self.playing_color

            #PGN update
            self._update_PGN(start, end, filepath="AI/")
            self.nb_pieces = np.sum(self.board.board != None)

            turn_count+=1

        winner = 2*(1-self.playing_color)-1 if turn_count<max_turns else 0
        return torch.stack(game_states), torch.stack(piece_log_probs), torch.stack(end_log_probs), winner


class Button:
    def __init__(self, x:int, y:int, width:int, height:int, game:Game, function, args=None, text:str = '', color = (255, 255, 255), margin:float = 0.2):
        self.coords=np.array([x,y])
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.size = np.array([width, height])
        self.margin_size = self.size * margin
        self.game = game
        self.function = function
        self.args = args

        self.text = self.game.smallfont.render(self.text, True, (255, 255, 255))
        self.textrect = self.text.get_rect()
        self.textrect.center = self.coords + self.size/2

    
    def show(self):
        pygame.draw.rect(self.game.window, self.color, [self.coords[0], self.coords[1], self.width, self.height], 0)
        pygame.draw.rect(self.game.window, np.array(self.color) /1.5, [self.coords[0] + self.margin_size[0], self.coords[1]  + self.margin_size[1], self.width  -2* self.margin_size[0], self.height  -2* self.margin_size[1]], 0)
        self.game.window.blit(self.text, self.textrect)

    def trigger_if_pressed(self, coords):
        if np.all(coords >= self.coords) and np.all(coords <= self.coords + self.size): 
            self.function(*self.args)
            self.game.selected_piece = None


if __name__ == "__main__":
    game = Game()
    with open("PGN/AI/1762783942.0622911.pgn", 'r', encoding="utf-8") as f:
        pgnstring = f.read()

    game.load_PGN(pgnstring)
    game.play_human_human_game()

# TODO factorise game loop