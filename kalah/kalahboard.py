class KalahBoard:
    """
    A implementation of the game Kalah

    Example layout for a 6 bin board:
           <--- North
     ------------------------  
      12  11  10   9   8   7   
                               
      13                   6   
                               
       0   1   2   3   4   5   
     ------------------------  
             South --->        
    """
    
    bins = 0
    seeds = 0

    board = []
    player = 0

    def __init__(self, bins, seeds):
        self.bins = bins
        self.seeds = seeds

        self.board = [seeds]*(bins*2+2)
        self.board[bins] = 0
        self.board[2*bins + 1] = 0

        if not self._check_board_consistency(self.board):
            raise ValueError('The board created is not consistent, some error must have happened')

        self._player_houses = { 0: self.bins*(1),
                                1: self.bins*(2) + 1}

    def copy(self):
        board = KalahBoard(self.bins, self.seeds)

        board.set_board(list(self.board))
        board.set_current_player(self.player)

        if not board._check_board_consistency(board.board):
            raise ValueError('The board that was copied is not consistent, some error must have happened')

        return board

    def reset(self):
        self.board = [self.seeds]*(self.bins*2+2)
        self.board[self.bins] = 0
        self.board[2*self.bins + 1] = 0

        self.player = 0

    def pretty_print(self):
        print(self.get_board())

    def move(self, b):
        if b not in self.allowed_moves():
            return False

        old_board = list(self.board)

        seeds_to_distribute = self.board[b]
        self.board[b] = 0

        other_player = 1 if self.current_player() == 0 else 0

        current_bin = b
        while seeds_to_distribute > 0:
            current_bin = current_bin+1

            if current_bin >= len(self.board):
                current_bin -= len(self.board)

            if current_bin == self._get_house(other_player):
                continue

            self.board[current_bin] += 1
            seeds_to_distribute -= 1

        # Seed in empty bin -> take seeds on the opponents side
        if ( current_bin != self.get_house_id(self.current_player()) and
                self.board[current_bin] == 1 and
                current_bin >= self._get_first_bin(self.current_player()) and
                current_bin < self._get_last_bin(self.current_player()) ):
            opposite_bin = current_bin + self.bins+1
            if opposite_bin >= len(self.board):
                opposite_bin -= len(self.board)
            if self.board[opposite_bin] > 0:
                self.board[self._get_house(self.current_player())] += self.board[opposite_bin] + self.board[current_bin]
                self.board[opposite_bin] = 0
                self.board[current_bin] = 0

        # All seeds empty, opponent takes all his seeds
        if self._all_empty_bins(self.current_player()):
            for b in range(self.bins):
                self.board[self._get_house(other_player)] += self.board[other_player*self.bins + other_player + b]
                self.board[other_player*self.bins + other_player + b] = 0

        if current_bin != self.get_house_id(self.current_player()):
            self.player = 1 if self.current_player() == 0 else 0

        if not self._check_board_consistency(self.board):
            raise ValueError('The board is not consistent, some error must have happened. Old Board: ' + str(old_board) + ", move = " + str(b) +", new Board: " + str(self.get_board()))

        return True

    def _get_first_bin(self, player):
        return player*self.bins + player

    def _get_last_bin(self, player):
        return self._get_first_bin(player) + self.bins - 1

    def _all_empty_bins(self, player):
        player_board = self._get_player_board(player)
        for seed in player_board[:-1]:
            if seed > 0:
                return False

        return True

    def _check_board_consistency(self, board):
        expected_seeds = 2*self.seeds*self.bins

        actual_seeds = 0
        for s in board:
            actual_seeds += s

        return actual_seeds == expected_seeds

    def _get_house(self, player):
        return self._player_houses[player]

    def get_board(self):
        return list(self.board)

    def current_player(self):
        return self.player
    
    def _get_player_board(self, player):
        return self.get_board()[player*self.bins + player : player*self.bins + player + self.bins + 1]

    def game_over(self):
        player_board_one = self._get_player_board(0)
        player_board_two = self._get_player_board(1)

        player_one_empty = True
        for seed in player_board_one[:-1]:
            if seed > 0:
                player_one_empty = False

        player_two_empty = True
        for seed in player_board_two[:-1]:
            if seed > 0:
                player_two_empty = False

        return player_one_empty or player_two_empty

    def score(self):
        return [self.board[self.bins], self.board[2*self.bins + 1]]

    def allowed_moves(self):
        allowed_moves = []
        player_board = self._get_player_board(self.current_player())
        for b in range(len(player_board)-1):
            if player_board[b] > 0:
                allowed_moves.append(b + self.current_player()*self.bins + self.current_player())
        return allowed_moves

    def set_board(self, board):
        if len(board) != self.bins*2 + 2:
            raise ValueError('Passed board size does not match number of bins = ' + str(self.bins) + ' used to create the board')

        if not self._check_board_consistency(board):
            raise ValueError('The board is not consistent, cannot use it')

        self.board = list(board)

    def set_current_player(self, player):
        if player >= 0 and player < 2:
            self.player = player
        else:
            raise ValueError('Passed player number is not 0 or 1')

    def current_player_score(self):
        return self.score()[self.current_player()]

    def get_house_id(self, player):
        return self._get_house(player)
