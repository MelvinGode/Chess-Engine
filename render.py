import pygame
import os
import chessgame
import numpy as np

pygame.init()

WIDTH, HEIGHT = 800, 800

# Grid
corner_x = 50
corner_y = 50
margin = np.array([corner_x, corner_y])
square_size = 70
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Chess Engine')
pygame.draw.rect(window, (255, 255, 255), [corner_x, corner_y, square_size*8, square_size*8], 1)
def draw_grid():
    for i in range(0, 64, 1):
        y = (i//8) * square_size +corner_y
        x = (i%8) * square_size +corner_x
        isgrey = (i%2)
        if (i//8)%2 : isgrey = 1-isgrey
        color =  (50,50,50) if isgrey else (255, 255, 255)
        pygame.draw.rect(window, color, [x, y, square_size, square_size], 0)

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

    whoseturnisit = font.render(f"{'White' if playing_color else 'Black'}\'s move", True, (255,255,255))
    rect = whoseturnisit.get_rect()
    rect.center = ((board.width * square_size)//2 + corner_x, board.height* square_size + corner_y + 100)
    window.blit(whoseturnisit, rect)

    gameover = False
    while running:
        while not gameover:
            current_player_legal_moves = board.get_all_legal_moves(playing_color)
            if not len(current_player_legal_moves): 
                gameover=True
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running=False
                    gameover = True
                    break

                if event.type == pygame.MOUSEBUTTONDOWN:
                    board_coords = (np.array(pygame.mouse.get_pos()) - margin) // square_size

                    # If we are already holding a piece in hand
                    if selected_piece is not None:
                        try: 
                            illegal_flag = board.move([selected_piece.x, selected_piece.y], board_coords)
                            if illegal_flag: 
                                selected_piece = None
                                continue
                        except Exception as e:
                            print(f"Error {e}")
                            selected_piece = None
                            continue
                        playing_color = 1-playing_color
                        selected_piece = None

                        window.fill(0)
                        whoseturnisit = font.render(f"{'White' if playing_color else 'Black'}\'s move", True, (255,255,255))
                        window.blit(whoseturnisit, rect)

                        display_board(board)
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
                        highlight_color = (100, 100, 255)
                        pygame.draw.rect(window, highlight_color, [xy_coords[0], xy_coords[1], square_size, square_size], 2)
                        # pygame.draw.rect(window, (255, 255, 255), [xy_coords[0], xy_coords[1], square_size, square_size], 3)

                pygame.display.update()

        winnertext = font.render(f"Checkmate, {'White' if 1-playing_color else 'Black'} wins", True, (255,255,255))
        rect = winnertext.get_rect()
        rect.center = ((board.width * square_size)//2 + corner_x, board.height* square_size + 2*corner_y)
        window.blit(winnertext, rect)

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