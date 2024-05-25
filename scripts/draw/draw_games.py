from chessml import script
from chessml.models.lightning.conv_vae import ConvVAE
from chessml.data.games.games_from_pgn import GamesFromPGN
from chessml.data.boards.boards_from_games import BoardsFromGames
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from chessml.data.boards.board_representation import BoardRepresentation, OnlyPieces
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from lightning import LightningModule


def plot_trajectories(trajectories):
    plt.figure(figsize=(10, 6))

    # Define the color map
    cmap = plt.get_cmap("viridis")

    # Loop through trajectories and plot each one
    for i, traj in enumerate(trajectories):
        # algo = TSNE(2, random_state=42)
        algo = PCA(n_components=2)
        traj = algo.fit_transform(traj)

        x, y = traj[:, 0], traj[:, 1]
        color = cmap(float(i) / len(trajectories))
        plt.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            markersize=6,
            linewidth=2,
            color=color,
            label=f"Trajectory {i + 1}",
        )

    # Customize the plot
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Trajectories")
    plt.legend()
    plt.grid(True)

    # Show the plot
    # plt.show()
    plt.savefig("2d.png", dpi=300, bbox_inches="tight")


def trajectories_generator(
    games: GamesFromPGN,
    board_representation: BoardRepresentation,
    model: LightningModule,
):
    for game in games:
        board_tensors = torch.as_tensor(
            list(BoardsFromGames(games=[game], transforms=[board_representation],))
        )

        with torch.no_grad():
            trajectory = model(board_tensors)

        yield trajectory


@script
def main(args, config):
    games = GamesFromPGN(
        path=Path("./datasets/championships/WorldChamp2021.pgn"), limit=3
    )

    autoencoder = ConvVAE.load_from_checkpoint(
        "./checkpoints/board_vae-ld=512-ks=2-cm=2-step=1450.ckpt"
    )
    autoencoder.eval()

    board_representation = OnlyPieces()

    trajectories = list(
        trajectories_generator(games, board_representation, autoencoder)
    )

    plot_trajectories(trajectories)


# from chessml import script
# from chessml.data.games.games_from_pgn import GamesFromPGN
# from pathlib import Path
# from chess.pgn import Game
# from argparse import Namespace
# from chess import Board
# import torch
# import numpy as np
# from sklearn.manifold import TSNE
# import plotly.express as px
# from chessml.utils import numpy_to_batched_tensor


# def main(args: Namespace):
#     """
#     Takes games from PGN file and draws their trajectories in 2D
#     """

#     with torch.no_grad():

#         def filter(game: Game):
#             if game.headers["Event"] == "SuperGM":
#                 return True

#             return False

#         games = GamesFromPGN(
#             path=Path("./datasets/players/Kasparov.pgn"),
#             filter=filter,
#         )

#         autoencoder = PositionAutoencoder.load_from_checkpoint(
#             "./checkpoints/epoch=16-step=53125.ckpt"
#         )
#         autoencoder.eval()

#         board = Board()

#         data = {
#             "round": [],
#             "move": [],
#             "position": [],
#             "result": [],
#         }

#         starting_position_array = board_to_position_array(board)

#         for game in games:
#             board.reset()

#             data["round"].append(game["round"])
#             data["result"].append(game["result"].name)
#             data["move"].append("@")
#             data["position"].append(starting_position_array)

#             for move in game["moves"]:
#                 board.push_san(move)
#                 array = board_to_position_array(board)

#                 data["round"].append(game["round"])
#                 data["result"].append(game["result"].name)
#                 data["move"].append(move)
#                 data["position"].append(array)

#         tensor = torch.from_numpy(np.array(data["position"]))
#         print(f"{tensor.shape=}")

#         latent = autoencoder.encode(tensor)
#         print(f"{latent.shape=}")

#         tsne = TSNE(2)
#         tsne_result = tsne.fit_transform(latent)
#         print(f"{tsne_result.shape=}")

#         data["x"] = tsne_result[:, 0]
#         data["y"] = tsne_result[:, 1]

#         fig = px.line(
#             data,
#             x="x",
#             y="y",
#             color="round",
#             symbol="round",
#         )

#         fig.show()


# script.run(main)
