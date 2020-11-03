import argparse
import time

from kalah.kalahboard import KalahBoard

from kalah.agents.randomagent import RandomAgent
from kalah.agents.maxscoreagent import MaxScoreAgent
from kalah.agents.maxscorerepeatagent import MaxScoreRepeatAgent
from kalah.agents.minimaxagent import MinimaxAgent

from kalah.kalahbattleground import KalahBattleground

parser = argparse.ArgumentParser(description='Comparison of classic agents for Kalah')
parser.add_argument('--bins', type=int, default=6, metavar='B',
                    help='bins of the Kalah board (default: 6)')
parser.add_argument('--seeds', type=int, default=4, metavar='S',
                    help='seeds of the Kalah board (default: 4)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--games', type=int, default=1000, metavar='N',
                    help='number of games for each comparison (default: 1000)')
parser.add_argument('--minimax_depth', type=int, default=4, metavar='D',
                    help='max depth of the minimax agent (default: 4)')
args = parser.parse_args()

def main():
    n_games = args.games

    agent_classes = [RandomAgent, MaxScoreAgent, MaxScoreRepeatAgent, MinimaxAgent]

    battleground = KalahBattleground(args.bins, args.seeds)

    for agent_class1 in agent_classes:
        for agent_class2 in agent_classes:
            results = battleground.battle(lambda seed: agent_class1(seed), lambda seed: agent_class2(seed), args.games, args.seed)

            if results.draws != n_games:
                print(agent_class1.__name__, "won", 100*results.wins_agent1 / results.n_games,
                    "% of all N =", results.n_games ,"games against", agent_class2.__name__,
                    "Number of draws:", results.draws)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
