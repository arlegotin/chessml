from chessml import script, config
import os
import requests
from pathlib import Path


def download_chess_images(source_name, download_url_template, piece_sets, dist):
    piece_name_map = {
        "bp": "Pawn",
        "bb": "Bishop",
        "bn": "Knight",
        "br": "Rook",
        "bq": "Queen",
        "bk": "King",
        "wp": "Pawn",
        "wb": "Bishop",
        "wn": "Knight",
        "wr": "Rook",
        "wq": "Queen",
        "wk": "King",
    }

    for piece_set in piece_sets:
        piece_set_path = os.path.join(dist, f"{source_name}_{piece_set}")
        os.makedirs(piece_set_path, exist_ok=True)

        for piece_code, piece_name in piece_name_map.items():
            color = "Black" if piece_code[0] == "b" else "White"
            color_path = os.path.join(piece_set_path, color)
            os.makedirs(color_path, exist_ok=True)

            file_name = f"{piece_name}.png"
            file_path = os.path.join(color_path, file_name)

            url = download_url_template.format(piece_set=piece_set, piece_code=piece_code)
            print(url, file_path)
            quit()

            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
            else:
                print(f"Failed to download {url}")


@script
def main(args):
    for section in config.assets.piece_sets:
        download_chess_images(
            section.name,
            section.download_url_template,
            section.sets,
            Path("assets/piece_png"),
        )
