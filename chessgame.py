import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy



class Piece():
    def __init__(self, x: int, y: int, name: str, color: int):
        self.x = x
        self.y = y
        self.color = color
        self.name = name
   

class Chessboard:
    
    def __init__(self, width:int, height:int, pieces):
        self.pieces = pieces

        self.width = width
        self.height = height
        self.board = np.full((width, height), None, dtype=object)
        for piece in self.pieces:
            self.board[piece.x, piece.y] = piece

    def get_piece_legal_moves(self, piece:Piece, depth:int=0):
        start_location = np.array([piece.x, piece.y])
        player_color = piece.color
        moves = [] # Array with dims nb_legal_moves*2*2 for start-finish coords and x-y
        if piece.name=="king":
            for x_offset in [-1, 0, 1]:
                for y_offset in [-1, 0, 1]:
                    if abs(x_offset) + abs(y_offset) == 0 : continue
                    end_location = start_location+np.array([x_offset, y_offset])

                    #Check if location is on board
                    if np.any(end_location < 0) or end_location[0]>= self.width or end_location[1]>=self.height:
                        continue
                    #Check if any allied piece is on that location:
                    if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color == player_color:
                        continue
                    # Special check for king : check if suicide
                    if depth==0:
                        fake_board = deepcopy(self)
                        fake_board.move(start_location, end_location)

                        opponent_moves = fake_board.get_all_legal_moves(1-player_color, depth=1)
                        if np.any(np.logical_and(end_location[0]==opponent_moves[:,1,0], end_location[1]==opponent_moves[:,1,1])): 
                            print("forbid", end_location)
                            continue # If the opponent can move a piece to the future location of the king, forbid the move

                    moves.append([start_location, end_location])
            #TODO add rock

        if piece.name=="knight":
            end_locations = np.array([[1, 2], [1, -2], [2, 1], [2,-1], [-2, 1], [-2, -1], [-1, 2], [-1, -2]]) + start_location
            for end_location in end_locations:
                #Check if location is on board
                if np.any(end_location < 0) or end_location[0]>= self.width or end_location[1]>=self.height:continue
                #Check if any allied piece is on that location:
                if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color == player_color:continue
                moves.append([start_location, end_location])

        if piece.name=="pawn":
            end_locations = []
            color_multiplier = (2*(1-player_color)-1)
            end_locations.append(start_location + np.array([0, color_multiplier])) # Straight ahead
            if start_location[1] == player_color * (self.height-1) + color_multiplier: # Pawns on the starting row can move two squares ahead
                end_locations.append(start_location + np.array([0, 2*color_multiplier]))
            #diagonal kills
            for kill_location in start_location + np.array([[1, color_multiplier], [-1, color_multiplier]]):
                try: 
                    if self.board[kill_location[0], kill_location[1]] is not None and self.board[kill_location[0], kill_location[1]].color != player_color:
                        moves.append([start_location, kill_location])
                except Exception: pass # If the pawn is on the edge of the board and thus will trigger out of bounds error
            # Fuck en-passant, all my homies hate en-passant

            for end_location in end_locations:
                if np.any(end_location < 0) or end_location[0]>= self.width or end_location[1]>=self.height:continue #Check if location is on board
                if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color == player_color:continue #Check if any allied piece is on that location:
                moves.append([start_location, end_location])

            if end_location[1] == player_color * (self.height-1): # Promotion
                # TODO handle promotion TODO do it in move() function
                pass
        
        if piece.name in ["jester", "bishop"]:
            for x_offset in [-1, 1]:
                for y_offset in [-1, 1]:
                    if abs(x_offset) + abs(y_offset) == 0 : continue # cant just stay in place
                    previous_end_location = start_location
                    while True:
                        end_location = previous_end_location + np.array([x_offset, y_offset])
                        if np.any(end_location < 0) or end_location[0]>= self.width or end_location[1]>=self.height:break #Check if location is on board
                        if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color == player_color:break #Check if any allied piece is on that location:
                        if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color != player_color: #Check if an enemy piece is on that location
                            moves.append([start_location, end_location])
                            break
                        moves.append([start_location, end_location])
                        previous_end_location = end_location
        
        if piece.name in ["rook", "tower"]:
            for x_offset in [0, 1, -1]:
                for y_offset in [0, -1, 1]:
                    if abs(x_offset) + abs(y_offset) != 1 : continue # Move exactly one coordinate
                    previous_end_location = start_location
                    while True:
                        end_location = previous_end_location + np.array([x_offset, y_offset])
                        if np.any(end_location < 0) or end_location[0]>= self.width or end_location[1]>=self.height:break #Check if location is on board
                        if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color == player_color:break #Check if any allied piece is on that location:
                        if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color != player_color: #Check if an enemy piece is on that location
                            moves.append([start_location, end_location])
                            break
                        moves.append([start_location, end_location])
                        previous_end_location = end_location

        if piece.name=="queen":
            for x_offset in [0, 1, -1]:
                for y_offset in [0, -1, 1]:
                    if abs(x_offset) + abs(y_offset) == 0 : continue # cant just stay in place
                    previous_end_location = start_location
                    while True:
                        end_location = previous_end_location + np.array([x_offset, y_offset])
                        if np.any(end_location < 0) or end_location[0]>= self.width or end_location[1]>=self.height:break #Check if location is on board
                        if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color == player_color:break #Check if any allied piece is on that location:
                        if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color != player_color: #Check if an enemy piece is on that location
                            moves.append([start_location, end_location])
                            break
                        moves.append([start_location, end_location])
                        previous_end_location = end_location
        
        moves = np.array(moves)
        if len(moves) == 0 : return np.empty((0, 2, 2))
        return moves

                        
    def get_all_legal_moves(self, player_color: int, depth:int=0):
        moves = [] # Array with dims nb_legal_moves*2*2 for start-finish coords and x-y
        for piece in self.board.ravel():
            if piece is None or piece.color == player_color : continue
            moves.extend(self.get_piece_legal_moves(piece, depth=depth))
        return np.array(moves)
    
    def move(self, start, end):
        #Check if move is legal
        selected_piece = self.board[start[0], start[1]]
        if selected_piece is None : return
        legal_moves = self.get_piece_legal_moves(selected_piece, depth=1)[:,1,:]
        if not np.any(np.logical_and(end[0]==legal_moves[:,0], end[1]==legal_moves[:,1])): return

        self.board[end[0], end[1]] = selected_piece
        self.board[start[0], start[1]] = None
        selected_piece.x = end[0]
        selected_piece.y = end[1]

    
def create_classic_board():
    pieces = []
    for i, name in enumerate(["rook", "knight", "bishop"]):
        pw = Piece(x=i, y=7, name=name, color=1)
        pb = Piece(x=i, y=0, name=name, color=0)
        pw2 = Piece(x=7-i, y=7, name=name, color=1)
        pb2 = Piece(x=7-i, y=0, name=name, color=0)
        pieces.extend([pw,pb,pw2,pb2])
    
    kingw = Piece(x=4, y=7, name="king", color=1)
    kingb = Piece(x=4, y=0, name="king", color=0)
    queenb = Piece(x=3, y=0, name="queen", color=0)
    queenw = Piece(x=3, y=7, name="queen", color=1)
    pieces.extend([kingw, kingb, queenb, queenw])
    for i in range(8):
        pawnw = Piece(x=i, y=6, name="pawn", color=1)
        pawnb = Piece(x=i, y=1, name="pawn", color=0)
        pieces.extend([pawnw, pawnb])
    
    board = Chessboard(8, 8, pieces)
    return board



