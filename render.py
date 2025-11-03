import pygame
import os
import chessgame
import numpy as np
from copy import deepcopy
from pygame import mixer

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
pygame.init()

WIDTH, HEIGHT = 1000, 800

class Game:
    def __init__(self):
        self.board = chessgame.create_classic_board()
        self.check = [False, False]
        self.gameover = False



# Grid
corner_x = 50
corner_y = 50
margin = np.array([corner_x, corner_y])
square_size = 70
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Chess Engine')
def draw_grid():
    for i in range(0, 64, 1):
        y = (i//8) * square_size +corner_y
        x = (i%8) * square_size +corner_x
        isgrey = (i%2)
        if (i//8)%2 : isgrey = 1-isgrey
        color =  (50,50,50) if isgrey else (255, 255, 255)
        pygame.draw.rect(window, color, [x, y, square_size, square_size], 0)
    pygame.draw.rect(window, (255, 255, 255), [corner_x, corner_y, square_size*8, square_size*8], 1)


# Cell texts
pygame.font.get_init()
font = pygame.font.SysFont('freesanbold.ttf', 50)
smallfont = pygame.font.SysFont('freesanbold.ttf', 30)
def dispay_coords():
    texts = []
    coords = []
    for i, letter in zip(range(8), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']):
        texts.append(font.render(letter, True, (255, 255, 255)))
        coords.append([corner_x + i*square_size + 35,  corner_y + square_size*8 + 35])

        texts.append(font.render(str(i+1), True, (255, 255, 255)))
        coords.append([corner_x //2, corner_y + (7-i)*square_size + 35])

    for text, coord in zip(texts, coords):
        textrect = text.get_rect()
        textrect.center = (coord[0], coord[1])
        window.blit(text, textrect)

#Pieces display
board = chessgame.create_classic_board()
os.chdir(os.path.dirname(os.path.realpath(__file__)))
piece_images = {}
for img_name in os.listdir("Assets/Pieces/"):
    piece_images[img_name[:-4]] = pygame.transform.scale_by(pygame.image.load(f"Assets/Pieces/{img_name}"), 1/2)
def display_pieces(board : chessgame.Chessboard):
    for piece in board.board.ravel():
        if piece is None: continue
        color = "white" if piece.color else 'black'
        sprite = piece_images[color+'-'+piece.name]
        coords = np.array([piece.x, piece.y])*square_size + square_size//2 + margin
        rect = sprite.get_rect()
        rect.center = (coords[0], coords[1])
        window.blit(sprite, rect)

#Sound
mixer.init()
mixer.Channel(1).set_volume(0.5)

def display_board(board):
    draw_grid()
    display_pieces(board)
    dispay_coords()
    reset_button.show()

class Button:
    def __init__(self, x:int, y:int, width:int, height:int, text:str = '', color = (255, 255, 255), margin:float = 0.2):
        self.coords=np.array([x,y])
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.size = np.array([width, height])
        self.margin_size = self.size * margin

        self.text = smallfont.render(self.text, True, (255, 255, 255))
        self.textrect = self.text.get_rect()
        self.textrect.center = self.coords + self.size/2

    
    def show(self):
        pygame.draw.rect(window, self.color, [self.coords[0], self.coords[1], self.width, self.height], 0)
        pygame.draw.rect(window, np.array(self.color) /2, [self.coords[0] + self.margin_size[0], self.coords[1]  + self.margin_size[1], self.width  -2* self.margin_size[0], self.height  -2* self.margin_size[1]], 0)
        window.blit(self.text, self.textrect)

reset_button_coords = margin + np.array([square_size * 8, 30]) + np.array([100, 0])
reset_button_size = np.array([170, 60])
reset_button = Button(reset_button_coords[0], reset_button_coords[1], reset_button_size[0], reset_button_size[1], text="New Game", color=(150, 50, 50), margin = .1)

def play_human_human_game():
    board = chessgame.create_classic_board()
    nb_pieces = np.sum(board.board != None)
    playing_color = 1
    running=True
    selected_piece = None
    check = [False, False]

    display_board(board)
    whoseturnisit = font.render(f"{'White' if playing_color else 'Black'}\'s move", True, (255,255,255))
    turn_rect = whoseturnisit.get_rect()
    turn_rect.center = ((board.width * square_size)//2 + corner_x, board.height* square_size + corner_y + 100)
    window.blit(whoseturnisit, turn_rect)
    pygame.display.update()

    gameover = False
    while running:
        while not gameover:
            for event in pygame.event.get():
                if gameover: 
                    break

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_coords = np.array(pygame.mouse.get_pos())
                    if np.all(mouse_coords >= reset_button.coords) and np.all(mouse_coords <= reset_button.coords + reset_button.size): # If we click on reset button
                        # Reset the game
                        board = chessgame.create_classic_board()
                        nb_pieces = np.sum(board.board != None)
                        selected_piece = None
                        playing_color = 1
                        mixer.Channel(1).play(pygame.mixer.Sound("Assets/Sounds/reset.mp3"))


                    board_coords = (mouse_coords - margin) // square_size

                    # If we are already holding a piece in hand
                    if selected_piece is not None:
                        try: 
                            previous_coords = [selected_piece.x, selected_piece.y]
                            illegal_flag = board.move([selected_piece.x, selected_piece.y], board_coords)
                            if illegal_flag: 
                                selected_piece = None
                                display_board(board)
                                if illegal_flag==2: print("Suicide move, illegal")
                                continue
                        except Exception as e:
                            print(f"Error {e}")
                            selected_piece = None
                            display_board(board)
                            continue
                        
                        print(f'{str(selected_piece).replace("at", "to")} from {letters[previous_coords[0]]}{8-previous_coords[1]}')

                        if np.sum(board.board != None) < nb_pieces:
                            nb_pieces -= 1
                            mixer.Channel(0).play(pygame.mixer.Sound("Assets/Sounds/capture.mp3"))
                        else:
                            mixer.Channel(0).play(pygame.mixer.Sound("Assets/Sounds/move.mp3"))
                        #This is where we change turns 
                        playing_color = 1-playing_color
                        selected_piece = None

                        window.fill(0)
                        whoseturnisit = font.render(f"{'White' if playing_color else 'Black'}\'s move", True, (255,255,255))
                        window.blit(whoseturnisit, turn_rect)
                        display_board(board)

                        #mate checker
                        next_moves = board.get_all_legal_moves(playing_color)
                        gameover = True
                        for move in next_moves: # Check if next player has at least one legal move
                            fake_board = deepcopy(board)
                            illegality_flag = fake_board.move(move[0], move[1])
                            if not illegality_flag:
                                gameover = False
                                break

                        #check checker
                        # if check[playing_color]: 
                        #     check[playing_color] = False # You cannot have stayed in check on your opponent's turn
                        #     continue
                        # opponent_moves = board.get_all_legal_moves(1-playing_color)
                        # for landing in next_moves[:,1,:]:
                        #     if board.board[landing[0], landing[1]].name=="king" and board.board[landing[0], landing[1]].color != playing_color: #Check for color needed because castling can land a piece on where king currently is
                        #         white_check = True
                    
                    else:# If we just picked up a piece
                        display_board(board)
                        #convert x y coords in board coords
                        try: selected_piece = board.board[board_coords[0], board_coords[1]]
                        except Exception: continue
                        if selected_piece is None or selected_piece.color != playing_color: 
                            selected_piece = None
                            continue
                        # print(f"Selected {'white' if selected_piece.color else 'black'} {selected_piece.name}")
                        moves = board.get_piece_legal_moves(selected_piece)
                        end_coords = moves[:, 1, :]
                        if len(end_coords)==0:
                            selected_piece = None
                        # color legal squares
                        for board_coords in end_coords:
                            xy_coords = board_coords * square_size + margin
                            highlight_color = (255, 100, 100)
                            pygame.draw.rect(window, highlight_color, [xy_coords[0], xy_coords[1], square_size, square_size], 2)
                            # pygame.draw.rect(window, (255, 255, 255), [xy_coords[0], xy_coords[1], square_size, square_size], 3)

                if event.type == pygame.QUIT:
                    running=False
                    gameover = True

                pygame.display.update()
        
        for event in pygame.event.get():
            if gameover:
                    window.fill(0)
                    display_board(board)
                    winnertext = font.render(f"Checkmate, {'White' if 1-playing_color else 'Black'} wins", True, (255,255,255))
                    rect = winnertext.get_rect()
                    rect.center = ((board.width * square_size)//2 + corner_x, board.height* square_size + corner_y + 100)
                    window.blit(winnertext, rect)
                    pygame.display.update()
            if event.type == pygame.QUIT:
                running=False

        
play_human_human_game()

# selected_piece = None
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.MOUSEBUTTONDOWN:
#             board_coords = (np.array(pygame.mouse.get_pos()) - margin) // square_size

#             # If we are already holding a piece in hand
#             if selected_piece is not None:
#                 try: board.move([selected_piece.x, selected_piece.y], board_coords)
#                 except Exception as e:
#                     print(f"Error {e}")
#                     selected_piece = None
#                     continue
#                 selected_piece = None
#                 display_board(board)
#                 continue
            
#             # If we just picked up a piece
#             display_board(board)
#             #convert x y coords in board coords
#             try: selected_piece = board.board[board_coords[0], board_coords[1]]
#             except Exception: continue
#             if selected_piece is None : continue
#             print(f"Selected {'white' if selected_piece.color else 'black'} {selected_piece.name}")
#             moves = board.get_piece_legal_moves(selected_piece)
#             end_coords = moves[:, 1, :]
#             if len(end_coords)==0:
#                 selected_piece = None
#             # color legal squares
#             for board_coords in end_coords:
#                 xy_coords = board_coords * square_size + margin
#                 highlight_color = (200, 100, 100)
#                 pygame.draw.rect(window, highlight_color, [xy_coords[0], xy_coords[1], square_size, square_size], 0)
#                 pygame.draw.rect(window, (255, 255, 255), [xy_coords[0], xy_coords[1], square_size, square_size], 3)

#         if event.type == pygame.QUIT:
#             running=False
    
#     pygame.display.update()