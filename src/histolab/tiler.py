from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .slide import Slide
from .tile import Tile
from .types import CoordinatePair
from .util import lru_cache, resize_mask, scale_coordinates


class Tiler(ABC):
    @abstractmethod
    def extract(self, slide: Slide):
        raise NotImplementedError


class RandomTiler(Tiler):
    """Extractor of random tiles from a Slide, at the given level, with the given size.

    Attributes
    ----------
    tile_size : tuple of int
        (width, height) of the extracted tiles.
    n_tiles : int
        Maximum number of tiles to extract.
    level : int
        Level from which extract the tiles. Default is 0.
    seed : int
        Seed for RandomState. Must be convertible to 32 bit unsigned integers. Default
        is 7.
    check_tissue : bool
        Whether to check if the tile has enough tissue to be saved. Default is True.
    prefix : str
        Prefix to be added to the tile filename. Default is an empty string.
    suffix : str
        Suffix to be added to the tile filename. Default is '.png'
    max_iter : int
        Maximum number of iterations performed when searching for eligible (if
        ``check_tissue=True``) tiles. Must be grater than or equal to ``n_tiles``.
    """

    def __init__(
        self,
        tile_size: Tuple[int, int],
        n_tiles: int,
        level: int = 0,
        seed: int = 7,
        check_tissue: bool = True,
        prefix: str = "",
        suffix: str = ".png",
        max_iter: int = 1e4,
    ):
        """RandomTiler constructor.

        Parameters
        ----------
        tile_size : tuple of int
            (width, height) of the extracted tiles.
        n_tiles : int
            Maximum number of tiles to extract.
        level : int
            Level from which extract the tiles. Default is 0.
        seed : int
            Seed for RandomState. Must be convertible to 32 bit unsigned integers.
            Default is 7.
        check_tissue : bool
            Whether to check if the tile has enough tissue to be saved. Default is True.
        prefix : str
            Prefix to be added to the tile filename. Default is an empty string.
        suffix : str
            Suffix to be added to the tile filename. Default is '.png'
        max_iter : int
            Maximum number of iterations performed when searching for eligible (if
            ``check_tissue=True``) tiles. Must be grater than or equal to ``n_tiles``.
        """

        super().__init__()

        assert (
            max_iter >= n_tiles
        ), "The maximum number of iterations must be grater than or equal to the "
        f"maximum number of tiles. Got max_iter={max_iter} and n_tiles={n_tiles}."

        self.tile_size = tile_size
        self.max_iter = max_iter
        self.level = level
        self.n_tiles = n_tiles
        self.seed = seed
        self.check_tissue = check_tissue
        self.prefix = prefix
        self.suffix = suffix

    @lru_cache(maxsize=100)
    def box_mask(self, slide: Slide) -> np.ndarray:
        """Return binary mask at level 0 of the box to consider for tiles extraction.

        If `check_tissue` attribute is True, the mask pixels set to True will be the
        ones corresponding to the tissue box. Otherwise, all the mask pixels will be set
        to True.

        Parameters
        ----------
        slide : Slide
            The Slide from which to extract the extraction mask

        Returns
        -------
        np.ndarray
            Extraction mask at level 0
        """

        if self.check_tissue:
            return slide.biggest_tissue_box_mask
        else:
            return np.ones(slide.dimensions[::-1], dtype="bool")

    @lru_cache(maxsize=100)
    def box_mask_lvl(self, slide: Slide) -> np.ndarray:
        """Return binary mask at target level of the box to consider for the extraction.

        If ``check_tissue`` attribute is True, the mask pixels set to True will be the
        ones corresponding to the tissue box. Otherwise, all the mask pixels will be set
        to True.

        Parameters
        ----------
        slide : Slide
            The Slide from which to extract the extraction mask

        Returns
        -------
        np.ndarray
            Extraction mask at target level
        """

        box_mask_wsi = self.box_mask(slide)

        if self.level != 0:
            return resize_mask(
                box_mask_wsi, target_dimensions=slide.level_dimensions(self.level),
            )
        else:
            return box_mask_wsi

    def extract(self, slide: Slide):
        """Extract tiles consuming `random_tiles_generator` and save them to disk,
        following this filename pattern:
        `{prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}`
        """

        np.random.seed(self.seed)

        random_tiles = self._random_tiles_generator(slide)

        tiles_counter = 0
        for tiles_counter, (tile, tile_wsi_coords) in enumerate(random_tiles):
            tile_filename = self._tile_filename(tile_wsi_coords, tiles_counter)
            tile.save(tile_filename)
            print(f"\t Tile {tiles_counter} saved: {tile_filename}")
        print(f"{tiles_counter} Random Tiles have been saved.")

    def _random_tile_coordinates(self, slide: Slide) -> CoordinatePair:
        """Return 0-level Coordinates of a tile picked at random within the box.

        Parameters
        ----------
        slide : Slide
            Slide from which calculate the coordinates. Needed to calculate the box.

        Returns
        -------
        CoordinatePair
            Random tile Coordinates at level 0
        """
        box_mask_lvl = self.box_mask_lvl(slide)
        tile_w_lvl, tile_h_lvl = self.tile_size

        x_ul_lvl = np.random.choice(np.where(box_mask_lvl)[1])
        y_ul_lvl = np.random.choice(np.where(box_mask_lvl)[0])

        x_br_lvl = x_ul_lvl + tile_w_lvl
        y_br_lvl = y_ul_lvl + tile_h_lvl

        tile_wsi_coords = scale_coordinates(
            reference_coords=CoordinatePair(x_ul_lvl, y_ul_lvl, x_br_lvl, y_br_lvl),
            reference_size=slide.level_dimensions(level=self.level),
            target_size=slide.level_dimensions(level=0),
        )

        return tile_wsi_coords

    def _random_tiles_generator(self, slide: Slide) -> (Tile, CoordinatePair):
        """Generate Random Tiles within a slide box.

        If ``check_tissue`` attribute is True, the box corresponds to the tissue box,
        otherwise it corresponds to the whole level.

        Stops if:
        * the number of extracted tiles is equal to ``n_tiles`` OR
        * the maximum number of iterations ``max_iter`` is reached

        Parameters
        ----------
        slide : Slide
            The Whole Slide Image from which to extract the tiles.

        Yields
        ------
        tile : Tile
            The extracted Tile
        coords : CoordinatePair
            The level-0 coordinates of the extracted tile
        """

        iteration = valid_tile_counter = 0

        while True:

            iteration += 1

            tile_wsi_coords = self._random_tile_coordinates(slide)
            try:
                tile = slide.extract_tile(tile_wsi_coords, self.level)
            except ValueError:
                iteration -= 1
                continue

            if not self.check_tissue or tile.has_enough_tissue():
                yield tile, tile_wsi_coords
                valid_tile_counter += 1

            if self.max_iter and iteration > self.max_iter:
                break

            if valid_tile_counter > self.n_tiles:
                break

    def _tile_filename(self, tile_wsi_coords, tiles_counter):
        x_ul_wsi, y_ul_wsi, x_br_wsi, y_br_wsi = tile_wsi_coords
        tile_filename = (
            f"{self.prefix}tile_{tiles_counter}_level{self.level}_{x_ul_wsi}-{y_ul_wsi}"
            f"-{x_br_wsi}-{y_br_wsi}{self.suffix}"
        )

        return tile_filename


class GridTiler(Tiler):
    pass