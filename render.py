import pygame
import os
import chessgame
import numpy as np
from copy import deepcopy

pygame.init()

WIDTH, HEIGHT = 800, 800

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

def display_board(board):
    draw_grid()
    display_pieces(board)
    dispay_coords()

display_board(board)

pygame.display.update()
running = True

def play_human_human_game():
    board = chessgame.create_classic_board()
    playing_color = 1
    running=True
    selected_piece = None
    check = [False, False]

    whoseturnisit = font.render(f"{'White' if playing_color else 'Black'}\'s move", True, (255,255,255))
    turn_rect = whoseturnisit.get_rect()
    turn_rect.center = ((board.width * square_size)//2 + corner_x, board.height* square_size + corner_y + 100)
    window.blit(whoseturnisit, turn_rect)

    gameover = False
    while running:
        while not gameover:
            for event in pygame.event.get():
                if gameover: 
                    break

                if event.type == pygame.MOUSEBUTTONDOWN:
                    board_coords = (np.array(pygame.mouse.get_pos()) - margin) // square_size

                    # If we are already holding a piece in hand
                    if selected_piece is not None:
                        try: 
                            illegal_flag = board.move([selected_piece.x, selected_piece.y], board_coords)
                            if illegal_flag: 
                                selected_piece = None
                                display_board(board)
                                print("Suicide move, illegal")
                                continue
                        except Exception as e:
                            print(f"Error {e}")
                            selected_piece = None
                            display_board(board)
                            continue

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
                                print(f'Legal move :',move, "flag=",illegal_flag)
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
                        
                        continue
                    
                    # If we just picked up a piece
                    display_board(board)
                    #convert x y coords in board coords
                    try: selected_piece = board.board[board_coords[0], board_coords[1]]
                    except Exception: continue
                    if selected_piece is None or selected_piece.color != playing_color: 
                        selected_piece = None
                        continue
                    print(f"Selected {'white' if selected_piece.color else 'black'} {selected_piece.name}")
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