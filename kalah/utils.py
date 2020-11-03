from kalah.kalahboard import KalahBoard

def _get_score_gain(board: KalahBoard, player, mv):
    test_board = board.copy()
    old_score = test_board.score()[player]
    test_board.move(mv)
    return test_board.score()[player] - old_score

def get_best_moves(board: KalahBoard):
    best_moves = []
    best_score_gain = -1

    for mv in board.allowed_moves():
        score_gain = _get_score_gain(board, board.current_player(), mv)
        if score_gain > best_score_gain:
            best_moves = [mv]
            best_score_gain = score_gain
        elif score_gain == best_score_gain:
            best_moves.append(mv)

    return best_moves

def compare_agents(bins, seeds, n_games, agent1, agent2):
    wins_agent_one = 0
    wins_agent_two = 0
    draws = 0
    invalid_moves = 0

    current_game = 0
    while current_game < n_games:
        board = KalahBoard(bins, seeds)

        last_invalid_player = None
        invalid_count = 0
        while not board.game_over():
            if board.current_player() == 0:
                valid = board.move(agent1.get_next_move(board))
            else:
                valid = board.move(agent2.get_next_move(board))
            if not valid:
                if last_invalid_player == board.current_player():
                    invalid_count += 1
                else:
                    invalid_count = 0
                    last_invalid_player = board.current_player()
            if invalid_count > 10:
                break

        if invalid_count > 10:
            invalid_moves += 1
            if last_invalid_player == 0:
                wins_agent_two += 1
            else:
                wins_agent_one += 1
        else:
            if board.score()[0] > board.score()[1]:
                wins_agent_one += 1
            elif board.score()[0] < board.score()[1]:
                wins_agent_two += 1
            else:
                draws += 1

        current_game += 1

    return [wins_agent_one, wins_agent_two, draws, invalid_moves]
