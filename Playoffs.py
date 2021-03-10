import os
import pickle
import sys

from ArenaParallel import Arena
from gomaku.GomakuGame import GomakuGame
# from gomaku.GomakuPlayers import RandomPlayer
from utils import dotdict


def start_playoff(base_path, network_type, run_type, num_workers=1, worker_index=0):
    def get_run_path(network_type, run_type, index):
        return os.path.join(base_path, network_type, f"{run_type}-{index}")

    run_count = 1
    checkpoints = []
    while True:
        pth = get_run_path(network_type, run_type, run_count)
        if not os.path.exists(pth):
            break
        local_checkpoints = []
        for file in os.listdir(pth):
            if "checkpoint_" in file and ".examples" not in file:
                inter = file.split("_")[-1]
                num = int(inter.split(".")[0])
                local_checkpoints.append((num, file))
        local_checkpoints.sort()
        for checkpoint in local_checkpoints:
            checkpoints.append((run_count, *checkpoint))
        run_count += 1

    data_path = os.path.join(base_path, network_type, f"{run_type}.playoffs")
    global data
    data = {
        # (run_num, iteration, checkpoint): { (run_num, iteration, checkpoint): { stats: (wins, losses, draws) } }
    }

    def save_data(data):
        pickle.dump(data, open(data_path, "wb"))

    def load_data():
        return pickle.load(open(data_path, "rb"))

    if os.path.exists(data_path):
        data = load_data()

    print(checkpoints)

    def get_path(checkpoint):
        run, index, filename = checkpoint
        return os.path.join(base_path, network_type, f"{run_type}-{run}", filename)

    game = GomakuGame(8)

    def run_playoff(p1_checkpoint, p2_checkpoint, to=100):
        global data
        # Plays games between the two checkpoints until data has [to] games in it
        if p1_checkpoint not in data:
            data[p1_checkpoint] = {}
        data_p1 = data[p1_checkpoint]
        if p2_checkpoint not in data_p1:
            data_p1[p2_checkpoint] = {"stats": (0, 0, 0)}
        save_data(data)
        data_playoff = data_p1[p2_checkpoint]
        stats = data_playoff["stats"]
        data = load_data()  # Reload in case another worker already played this
        already_played = sum(stats)
        to_play = to - already_played
        if to_play > 0:
            print(f"Starting playoff between [Run: {p1_checkpoint[0]}, Checkpoint: {p1_checkpoint[1]}] and [Run: {p2_checkpoint[0]}, Checkpoint: {p2_checkpoint[1]}]")
            arena = Arena(get_path(p1_checkpoint), get_path(p2_checkpoint), 2, game, args=dotdict({'numMCTSSims': 100, 'cpuct': 1.0}), names=(f"w-{worker_index}-p1", f"w-{worker_index}-p2"))
            wins_p1, wins_p2, draws = arena.playGamesParallel(to_play, log=False)
            new_stats = (stats[0] + wins_p1, stats[1] + wins_p2, stats[2] + draws)
            data = load_data()  # Need to reload data in case a parallel worker wrote new data
            data[p1_checkpoint][p2_checkpoint]["stats"] = new_stats
            print(f"Stats: {new_stats}")
            save_data(data)  # Immediately save data so that other workers can update
            print(f"Saved: {data[p1_checkpoint]}")
        else:
            print(f"No games left to play for [Run: {p1_checkpoint[0]}, Checkpoint: {p1_checkpoint[1]}] and [Run: {p2_checkpoint[0]}, Checkpoint: {p2_checkpoint[1]}]")

    def playoffs(to=100):
        # Runs playoffs for every pair of checkpoints
        to_play = len(checkpoints)/num_workers
        start = int(round(to_play*worker_index))
        end = int(round(to_play*(worker_index+1)))
        global data
        for p1 in reversed(checkpoints):
            for p2 in checkpoints[start:end]:
                run_playoff(p1, p2, to)
                print("\n")

    playoffs(to=100)


if __name__ == "__main__":
    num_workers = 1
    worker_index = 0
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    if len(sys.argv) > 2:
        worker_index = int(sys.argv[2])
    if num_workers > 1:
        print(f"Starting playoffs with {num_workers} workers. I am worker {worker_index}.")

    base_path = "/content/drive/MyDrive/School/Hey, you're an engineer now/ESC190/"
    network_type = "OriginalNetwork"
    run_type = "100-20-Checkpoints"
    start_playoff(base_path, network_type, run_type, num_workers, worker_index)