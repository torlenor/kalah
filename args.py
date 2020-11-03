def add_common_train_args(parser):
    parser.add_argument('--episodes', type=int, default=1000, metavar='E',
                    help='number of episodes to train (default: 1000)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--bins', type=int, default=6, metavar='B',
                        help='bins of the Kalah board (default: 6)')
    parser.add_argument('--seeds', type=int, default=4, metavar='S',
                        help='seeds of the Kalah board (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=0.01, metavar='L',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--solved', type=float, default=95, metavar='SL',
                        help='consider problem solved when agent wins x percent of the games (default: 95)')
    parser.add_argument('--neurons', type=int, default=512, metavar='NE',
                        help='how many neurons in each layer (default: 512)')
    parser.add_argument('--evaluation-games', type=int, default=100, metavar='EG',
                        help='how many games to play to check win rate during training (default: 100)')
    parser.add_argument('--model-path', type=str, default='./model.pt', metavar='MP',
                        help='were to save the trained model (default: ./model.pt)')
