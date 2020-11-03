from kalah.kalahboard import KalahBoard

import unittest

# Unique board constelations to test:
#
# Normal move, no points
# Normal move, one seed in house
# Normal move, around the board, skip opponents house
# Hit own house, repeat move
# Hit own house after one full round around the board, repeat move
# Hit own empty bin, capture opponents and own seeds
# Hit own empty bin, but opponents bin empty, nothing should happen
# Hit enemy empty bin, nothing should happen
# End of game, opponent gets all his remaining seeds

class Test_TestKalahBoard(unittest.TestCase):
    def test_default_board(self):
        board = KalahBoard(6,4)
        self.assertEqual(board.get_board(), [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0])

        board = KalahBoard(9,6)
        self.assertEqual(board.get_board(), [6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0])

    def test_get_house(self):
        board = KalahBoard(2,2)
        self.assertEqual(board._get_house(0), 2)
        self.assertEqual(board._get_house(1), 5)

        board = KalahBoard(4,2)
        self.assertEqual(board._get_house(0), 4)
        self.assertEqual(board._get_house(1), 9)

        board = KalahBoard(4,4)
        self.assertEqual(board._get_house(0), 4)
        self.assertEqual(board._get_house(1), 9)

        board = KalahBoard(6,4)
        self.assertEqual(board._get_house(0), 6)
        self.assertEqual(board._get_house(1), 13)

        board = KalahBoard(6,6)
        self.assertEqual(board._get_house(0), 6)
        self.assertEqual(board._get_house(1), 13)

    def test_get_house_id(self):
        board = KalahBoard(2,2)
        self.assertEqual(board.get_house_id(0), 2)
        self.assertEqual(board.get_house_id(1), 5)

        board = KalahBoard(4,2)
        self.assertEqual(board.get_house_id(0), 4)
        self.assertEqual(board.get_house_id(1), 9)

        board = KalahBoard(4,4)
        self.assertEqual(board.get_house_id(0), 4)
        self.assertEqual(board.get_house_id(1), 9)

        board = KalahBoard(6,4)
        self.assertEqual(board.get_house_id(0), 6)
        self.assertEqual(board.get_house_id(1), 13)

        board = KalahBoard(6,6)
        self.assertEqual(board.get_house_id(0), 6)
        self.assertEqual(board.get_house_id(1), 13)

    def test_first_moves_6_4(self):
        board = KalahBoard(6,4)

        self.assertEqual(board.get_board(), [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0])
        self.assertEqual(board.current_player(), 0)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [0, 0])
        self.assertEqual(board.allowed_moves(), [0, 1, 2, 3, 4, 5])

        self.assertEqual(board.move(6), False)
        self.assertEqual(board.move(7), False)
        self.assertEqual(board.move(13), False)
        self.assertEqual(board.move(123), False)

        self.assertEqual(board.move(0), True)

        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [0, 0])
        self.assertEqual(board.allowed_moves(), [7, 8, 9, 10, 11, 12])

        self.assertEqual(board.move(7), True)

        self.assertEqual(board.current_player(), 0)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [0, 0])
        self.assertEqual(board.allowed_moves(), [1, 2, 3, 4, 5])

    def test_move_into_house_6_4(self):
        board = KalahBoard(6,4)

        self.assertEqual(board.get_board(), [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0])
        self.assertEqual(board.current_player(), 0)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [0, 0])
        self.assertEqual(board.allowed_moves(), [0, 1, 2, 3, 4, 5])

        self.assertEqual(board.move(2), True)

        self.assertEqual(board.get_board(), [4, 4, 0, 5, 5, 5, 1, 4, 4, 4, 4, 4, 4, 0])
        self.assertEqual(board.current_player(), 0)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [1, 0])
        self.assertEqual(board.allowed_moves(), [0, 1, 3, 4, 5])

        self.assertEqual(board.move(2), False)
        self.assertEqual(board.move(1), True)

        self.assertEqual(board.get_board(), [4, 0, 1, 6, 6, 6, 1, 4, 4, 4, 4, 4, 4, 0])
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [1, 0])
        self.assertEqual(board.allowed_moves(), [7, 8, 9, 10, 11, 12])

    def test_moves_6_4(self):
        board = KalahBoard(6,4)
        board.set_board([0, 0, 0, 0, 0, 1, 24, 0, 0, 0, 2, 0, 0, 21])
        board.set_current_player(1)

        self.assertEqual(board.get_board(), [0, 0, 0, 0, 0, 1, 24, 0, 0, 0, 2, 0, 0, 21])
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [24, 21])
        self.assertEqual(board.allowed_moves(), [10])

        self.assertEqual(board.move(10), True)

        self.assertEqual(board.get_board(), [0, 0, 0, 0, 0, 1, 24, 0, 0, 0, 0, 1, 1, 21])
        self.assertEqual(board.current_player(), 0)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [24, 21])
        self.assertEqual(board.allowed_moves(), [5])

        initial_board = [4, 4, 4, 4, 4, 0, 1, 5, 5, 5, 4, 4, 4, 0]
        board = KalahBoard(6,4)
        board.set_board(initial_board)
        board.set_current_player(1)

        self.assertEqual(board.get_board(), initial_board)
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [1, 0])
        self.assertEqual(board.allowed_moves(), [7, 8, 9, 10, 11, 12])

        self.assertEqual(board.move(8), True)

        self.assertEqual(board.get_board(), [4, 4, 4, 4, 4, 0, 1, 5, 0, 6, 5, 5, 5, 1])
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [1, 1])
        self.assertEqual(board.allowed_moves(), [7, 9, 10, 11, 12])


    def test_move_over_house_into_opponent_6_4(self):
        board = KalahBoard(6,4)

        self.assertEqual(board.get_board(), [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0])
        self.assertEqual(board.current_player(), 0)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [0, 0])
        self.assertEqual(board.allowed_moves(), [0, 1, 2, 3, 4, 5])

        self.assertEqual(board.move(5), True)

        self.assertEqual(board.get_board(), [4, 4, 4, 4, 4, 0, 1, 5, 5, 5, 4, 4, 4, 0])
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [1, 0])
        self.assertEqual(board.allowed_moves(), [7, 8, 9, 10, 11, 12])

    def test_end_game_collect_all_remaining_seeds_6_4(self):
        board = KalahBoard(6,4)

        board.set_board([0, 0, 1, 1, 0, 1, 30, 0, 0, 0, 0, 1, 0, 14])

        self.assertEqual(board.get_board(), [0, 0, 1, 1, 0, 1, 30, 0, 0, 0, 0, 1, 0, 14])
        self.assertEqual(board.current_player(), 0)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [30, 14])
        self.assertEqual(board.allowed_moves(), [2, 3, 5])

        self.assertEqual(board.move(2), True)

        self.assertEqual(board.get_board(), [0, 0, 0, 2, 0, 1, 30, 0, 0, 0, 0, 1, 0, 14])
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [30, 14])
        self.assertEqual(board.allowed_moves(), [11])

        self.assertEqual(board.move(11), True)

        self.assertEqual(board.get_board(), [0, 0, 0, 2, 0, 1, 30, 0, 0, 0, 0, 0, 1, 14])
        self.assertEqual(board.current_player(), 0)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [30, 14])
        self.assertEqual(board.allowed_moves(), [3, 5])

        self.assertEqual(board.move(3), True)

        self.assertEqual(board.get_board(), [0, 0, 0, 0, 1, 2, 30, 0, 0, 0, 0, 0, 1, 14])
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [30, 14])
        self.assertEqual(board.allowed_moves(), [12])

        self.assertEqual(board.move(12), True)

        self.assertEqual(board.get_board(), [0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 15])
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), True)
        self.assertEqual(board.score(), [33, 15])
        self.assertEqual(board.allowed_moves(), [])

    def test_end_game_collect_all_remaining_seeds_second_test_6_4(self):
        board = KalahBoard(6,4)

        board.set_board([0, 0, 0, 1, 1, 0, 24, 0, 0, 0, 0, 0, 1, 21])

        self.assertEqual(board.get_board(), [0, 0, 0, 1, 1, 0, 24, 0, 0, 0, 0, 0, 1, 21])
        self.assertEqual(board.current_player(), 0)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [24, 21])
        self.assertEqual(board.allowed_moves(), [3, 4])

        self.assertEqual(board.move(4), True)

        self.assertEqual(board.get_board(), [0, 0, 0, 1, 0, 1, 24, 0, 0, 0, 0, 0, 1, 21])
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [24, 21])
        self.assertEqual(board.allowed_moves(), [12])

        self.assertEqual(board.move(12), True)

        self.assertEqual(board.get_board(), [0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 22])
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), True)
        self.assertEqual(board.score(), [26, 22])
        self.assertEqual(board.allowed_moves(), [])

    def test_end_game_collect_all_remaining_seeds_third_test_2_2(self):
        board = KalahBoard(2,2)

        board.set_board([0, 3, 1, 2, 2, 0])

        self.assertEqual(board.get_board(), [0, 3, 1, 2, 2, 0])
        self.assertEqual(board.current_player(), 0)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [1, 0])
        self.assertEqual(board.allowed_moves(), [1])

        self.assertEqual(board.move(1), True)

        self.assertEqual(board.get_board(), [0, 0, 2, 0, 0, 6])
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), True)
        self.assertEqual(board.score(), [2, 6])
        self.assertEqual(board.allowed_moves(), [])

    def test_empty_pit_capture_4_4(self):
        # Test for player 1
        board = KalahBoard(4,4)
        board.set_current_player(0)
        board.set_board([1, 0, 4, 4, 7, 4, 4, 4, 4, 0])

        self.assertEqual(board.get_board(), [1, 0, 4, 4, 7, 4, 4, 4, 4, 0])
        self.assertEqual(board.current_player(), 0)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [7, 0])
        self.assertEqual(board.allowed_moves(), [0, 2, 3])

        self.assertEqual(board.move(0), True)

        self.assertEqual(board.get_board(), [0, 0, 4, 4, 12, 4, 0, 4, 4, 0])
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [12, 0])
        self.assertEqual(board.allowed_moves(), [5, 7, 8])

        # Test for player 2
        board = KalahBoard(4,4)
        board.set_current_player(1)
        board.set_board([4, 0, 5, 5, 1, 5, 4, 4, 4, 0])

        self.assertEqual(board.get_board(), [4, 0, 5, 5, 1, 5, 4, 4, 4, 0])
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [1, 0])
        self.assertEqual(board.allowed_moves(), [5, 6, 7, 8])

        self.assertEqual(board.move(7), True)

        self.assertEqual(board.get_board(), [5, 1, 5, 5, 1, 5, 4, 0, 5, 1])
        self.assertEqual(board.current_player(), 0)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [1, 1])
        self.assertEqual(board.allowed_moves(), [0, 1, 2, 3])
        

    def test_empty_pit_opposite_no_empty_capture_4_4(self):
        # We do not have the "empty capture" rule
        board = KalahBoard(4,4)

        board.set_board([1, 0, 4, 4, 7, 4, 0, 4, 4, 4])

        self.assertEqual(board.get_board(), [1, 0, 4, 4, 7, 4, 0, 4, 4, 4])
        self.assertEqual(board.current_player(), 0)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [7, 4])
        self.assertEqual(board.allowed_moves(), [0, 2, 3])

        self.assertEqual(board.move(0), True)

        self.assertEqual(board.get_board(), [0, 1, 4, 4, 7, 4, 0, 4, 4, 4])
        self.assertEqual(board.current_player(), 1)
        self.assertEqual(board.game_over(), False)
        self.assertEqual(board.score(), [7, 4])
        self.assertEqual(board.allowed_moves(), [5, 7, 8])

    def test_first_last_bin_functions(self):
        board = KalahBoard(4,4)

        self.assertEqual(board._get_first_bin(0), 0)
        self.assertEqual(board._get_last_bin(0), 3)
        
        self.assertEqual(board._get_first_bin(1), 5)
        self.assertEqual(board._get_last_bin(1), 8)

        board = KalahBoard(4,6)

        self.assertEqual(board._get_first_bin(0), 0)
        self.assertEqual(board._get_last_bin(0), 3)
        
        self.assertEqual(board._get_first_bin(1), 5)
        self.assertEqual(board._get_last_bin(1), 8)

        board = KalahBoard(2,4)

        self.assertEqual(board._get_first_bin(0), 0)
        self.assertEqual(board._get_last_bin(0), 1)
        
        self.assertEqual(board._get_first_bin(1), 3)
        self.assertEqual(board._get_last_bin(1), 4)

        board = KalahBoard(6,4)

        self.assertEqual(board._get_first_bin(0), 0)
        self.assertEqual(board._get_last_bin(0), 5)
        
        self.assertEqual(board._get_first_bin(1), 7)
        self.assertEqual(board._get_last_bin(1), 12)

if __name__ == '__main__':
    unittest.main()