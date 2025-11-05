import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

class Piece():
    def __init__(self, x: int, y: int, name: str, color: int):
        self.x = x
        self.y = y
        self.coords = np.array([x,y])
        self.color = color
        self.name = name
        self.hasmoved = False
        if name =="knight" : self.PGN_letter = "N"
        elif name == "queen": self.PGN_letter = "Q"
        elif name == "rook": self.PGN_letter = "R"
        elif name == "bishop": self.PGN_letter = "B"
        elif name == "pawn": self.PGN_letter = ""
        elif name == "king": self.PGN_letter = "K"
        else : raise ValueError("Piece name not accepted. Please choose one of 'pawn', 'rook', 'knigh', 'bishop', 'queen', 'king'.")

    def __str__(self):
        return f'{"white" if self.color else "black"} {self.name} at position {letters[self.x]}{8-self.y}'
   

class Chessboard:
    
    def __init__(self, width:int, height:int, pieces):
        self.pieces = pieces

        self.width = width
        self.height = height
        self.board = np.full((width, height), None, dtype=object)
        for piece in self.pieces:
            self.board[piece.x, piece.y] = piece

    def get_piece_legal_moves(self, piece:Piece, depth:int=0, collide_mode: bool = False):
        start_location = np.array([piece.x, piece.y])
        player_color = piece.color
        moves = [] # Array with dims nb_legal_moves*2*2 for start-finish coords and x-y
        if collide_mode : collisions = []
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
                        if collide_mode and self.board[end_location[0], end_location[1]].PGN_letter == piece.PGN_letter : collisions.append(end_location)
                        continue

                    moves.append([start_location, end_location])
            
            #Castling
            if not piece.hasmoved:
                #Kingside
                if self.board[7, piece.color * 7] is not None and not self.board[7, piece.color * 7].hasmoved:
                    if np.all(self.board[[5,6], piece.color * 7] == None):
                        moves.append([start_location, [start_location[0]+2, start_location[1]]])
                #Queenside
                if self.board[0, piece.color * 7] is not None and not self.board[0, piece.color * 7].hasmoved:
                    if np.all(self.board[[1,2,3], piece.color * 7] == None):
                        moves.append([start_location, [start_location[0]-2, start_location[1]]])

        if piece.name=="knight":
            end_locations = np.array([[1, 2], [1, -2], [2, 1], [2,-1], [-2, 1], [-2, -1], [-1, 2], [-1, -2]]) + start_location
            for end_location in end_locations:
                #Check if location is on board
                if np.any(end_location < 0) or end_location[0]>= self.width or end_location[1]>=self.height:continue
                #Check if any allied piece is on that location:
                if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color == player_color:
                    if collide_mode and self.board[end_location[0], end_location[1]].PGN_letter == piece.PGN_letter : collisions.append(end_location)
                    continue
                moves.append([start_location, end_location])

        if piece.name=="pawn":
            # TODO fix collide mode
            end_locations = []
            color_multiplier = (2*(1-player_color)-1)
            one_ahead = start_location + np.array([0, color_multiplier])
            two_ahead = start_location + np.array([0, 2*color_multiplier])
            if one_ahead[1]>=0 and one_ahead[1]<=self.height :
                if self.board[one_ahead[0], one_ahead[1]] is None:
                    moves.append([start_location, one_ahead])
                    if start_location[1] + collide_mode * color_multiplier*2 == player_color * (self.height-1) + collide_mode * color_multiplier * (self.height-1) + color_multiplier * (2*(1-collide_mode)-1) and two_ahead[1]>=0 and two_ahead[1]<=self.height :
                        if self.board[two_ahead[0], two_ahead[1]] is None :
                            moves.append([start_location, two_ahead])
                        elif collide_mode and self.board[two_ahead[0], two_ahead[1]].PGN_letter == piece.PGN_letter and self.board[two_ahead[0], two_ahead[1]].color != piece.color: collisions.append(two_ahead)
                elif collide_mode and self.board[one_ahead[0], one_ahead[1]].PGN_letter == piece.PGN_letter and self.board[one_ahead[0], one_ahead[1]].color != piece.color: collisions.append(one_ahead)

            #diagonal kills
            for kill_location in start_location + np.array([[1, color_multiplier], [-1, color_multiplier]]):
                try: 
                    if self.board[kill_location[0], kill_location[1]] is not None and self.board[kill_location[0], kill_location[1]].color != player_color:
                        moves.append([start_location, kill_location])
                        collisions.append(kill_location)
                except Exception: pass # If the pawn is on the edge of the board and thus will trigger out of bounds error
            # Fuck en-passant, all my homies hate en-passant
        
        if piece.name in ["jester", "bishop"]:
            for x_offset in [-1, 1]:
                for y_offset in [-1, 1]:
                    if abs(x_offset) + abs(y_offset) == 0 : continue # cant just stay in place
                    previous_end_location = start_location
                    while True:
                        end_location = previous_end_location + np.array([x_offset, y_offset])
                        if np.any(end_location < 0) or end_location[0]>= self.width or end_location[1]>=self.height:break #Check if location is on board
                        if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color == player_color: #Check if any allied piece is on that location:
                            if collide_mode and self.board[end_location[0], end_location[1]].PGN_letter == piece.PGN_letter : collisions.append(end_location)
                            break
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
                        if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color == player_color: #Check if any allied piece is on that location:
                            if collide_mode and self.board[end_location[0], end_location[1]].PGN_letter == piece.PGN_letter : collisions.append(end_location)
                            break
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
                        if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color == player_color: #Check if any allied piece is on that location:
                            if collide_mode and self.board[end_location[0], end_location[1]].PGN_letter == piece.PGN_letter : collisions.append(end_location)
                            break
                        if self.board[end_location[0], end_location[1]] is not None and self.board[end_location[0], end_location[1]].color != player_color: #Check if an enemy piece is on that location
                            moves.append([start_location, end_location])
                            break
                        moves.append([start_location, end_location])
                        previous_end_location = end_location
        
        if collide_mode : return np.array(collisions, dtype=int)
        moves = np.array(moves, dtype = int)
        if len(moves) == 0 : return np.empty((0, 2, 2))
        return moves

                        
    def get_all_legal_moves(self, player_color: int, depth:int=0):
        moves = [] # Array with dims nb_legal_moves*2*2 for start-finish coords and x-y
        for piece in self.board.ravel():
            if piece is None or piece.color != player_color : continue
            moves.extend(self.get_piece_legal_moves(piece, depth=depth))
        return np.array(moves)
    
    def move(self, start, end, depth:int=0):
        """Check if a move is legat and if so, commits the move on the board.
        Args:
            start: the coordinates of the moving piece
            end: the coordinated to move the piece to
        Returns 1 if move is purely illegal and hasn't been commited, 2 if a move is suicide and hasn't been comitted, 0 otherwise
        """
        #Check if move is legal
        selected_piece = self.board[start[0], start[1]]
        if selected_piece is None : return 1
        legal_moves = self.get_piece_legal_moves(selected_piece, depth=1)[:,1,:]
        if not np.any(np.logical_and(end[0]==legal_moves[:,0], end[1]==legal_moves[:,1])): return 1

        # Check if move exposes king
        if not depth:
            fakeboard = deepcopy(self)
            fakeboard.move(start,end, depth = 1)
            opponent_moves = fakeboard.get_all_legal_moves(1-selected_piece.color, depth=1)
            for opponent_end_position in opponent_moves[:, 1, :]:
                attacked_piece = fakeboard.board[opponent_end_position[0], opponent_end_position[1]]
                if attacked_piece is None: continue
                if attacked_piece.name=="king" and attacked_piece.color == selected_piece.color: 
                    return 2# Can't play move since it leads to checkmate
                
        #Castle check
        if selected_piece.name=="king" and abs(end[0] -start[0])>1:
            new_rook_x = start[0] + 2*(end[0]>start[0])-1
            castling_rook = self.board[(end[0]>start[0])*7, start[1]]
            self.board[castling_rook.x, start[1]] = None
            castling_rook.x = new_rook_x
            selected_piece.x = end[0]
            self.board[end[0], end[1]] = selected_piece
            self.board[new_rook_x, start[1]] = castling_rook
            self.board[start[0], start[1]] = None
            castling_rook.hasmoved = True
            selected_piece.hasmoved = True
            return 0 

        self.board[end[0], end[1]] = selected_piece
        self.board[start[0], start[1]] = None
        selected_piece.x = end[0]
        selected_piece.y = end[1]
        selected_piece.hasmoved = True

        # Pawn promotion
        if selected_piece.name == "pawn" and selected_piece.color*(7-selected_piece.y) + (1-selected_piece.color)*selected_piece.y == 7:
            selected_piece.name="queen"
            selected_piece.PGN_letter = 'Q'
        return 0

    
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

def create_fucked_up_board():
    pieces = []
    for i, name in enumerate(["rook", "knight", "bishop"]):
        pw = Piece(x=i, y=7, name=name, color=1)
        pw2 = Piece(x=7-i, y=7, name=name, color=1)
        pieces.extend([pw, pw2])
    
    kingw = Piece(x=4, y=7, name="king", color=1)
    kingb = Piece(x=4, y=0, name="king", color=0)
    queenw = Piece(x=3, y=7, name="queen", color=1)
    pieces.extend([kingw, queenw, kingb])
    
    board = Chessboard(8, 8, pieces)
    return board


#TODO LIST
# check handler
    # WIP
# Find a better way to detect suicide checks
# Make a proper Game class
# 
# Rewind move