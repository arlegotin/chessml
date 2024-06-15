from __future__ import annotations
from chessml import script, config
from pathlib import Path
from urllib.request import urlretrieve
from shutil import unpack_archive
from hashlib import md5
import logging
from tempfile import gettempdir


script.add_argument("-s", dest="section", default="players")


@script
def main(args):
    available_sections = config.dataset.sections

    if args.section not in available_sections:
        raise ValueError(f"there's no dataset section '{args.section}' in config")

    download_url_template = available_sections[args.section]["download_url_template"]
    names = available_sections[args.section]["names"]

    urls_to_download = map(lambda name: download_url_template.format(name=name), names)

    target_dir = Path(config.dataset.path) / args.section
    target_dir.mkdir(parents=True, exist_ok=True)

    for url in urls_to_download:
        url_path = Path(url)

        if url_path.suffix == ".zip":
            name = md5(url.encode("utf-8")).hexdigest()
            path_to_zip = Path(gettempdir()) / f"{name}.zip"

            logging.info(f"downloading {url} as {path_to_zip}")

            urlretrieve(url, path_to_zip)

            logging.info(f"unpacking {path_to_zip} to {target_dir}")
            unpack_archive(path_to_zip, target_dir)
            path_to_zip.unlink()

        elif url_path.suffix == ".pgn":
            path_to_pgn = target_dir / url_path.name
            logging.info(f"downloading {url} as {path_to_pgn}")
            urlretrieve(url, path_to_pgn)

        else:
            raise RuntimeError(f"invalid URL suffix {url_path.suffix}")
