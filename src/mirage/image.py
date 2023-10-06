from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Protocol, TypeVar, cast

from PIL import Image as PILImage, ImageFilter, ImageOps
import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import Tensor, device as Device, dtype as DType


@dataclass
class Mode(Protocol):
    name: str = ""
    channels: int = 0


@dataclass
class Grayscale(Mode):
    name = "L"
    channels = 1


@dataclass
class RGB(Mode):
    name = "RGB"
    channels = 3


@dataclass
class RGBA(Mode):
    name = "RGBA"
    channels = 4


MODES = [Grayscale, RGB, RGBA]

Point = tuple[int, int]
Box = tuple[Point, Point]
Color = tuple[int, int, int]

T = TypeVar("T", bound=Mode)
U = TypeVar("U", bound=Mode)


@dataclass(frozen=True)
class Image(Generic[T]):
    _pil: PILImage.Image
    _mode: T

    def __getitem__(self, key: int) -> "Image[Grayscale]":
        if key < 0:
            key = self._mode.channels + key
        assert 0 <= key <= self._mode.channels, f"key {key} out of bounds"
        return cast(
            Image[Grayscale], self._bind(lambda pil: pil.getchannel(channel=key))
        )

    @property
    def pil(self) -> PILImage.Image:
        return self._pil

    @property
    def mode(self) -> type[T]:
        return self._mode.__class__

    @property
    def channels(self) -> list["Image[Grayscale]"]:
        return [self[i] for i in range(self._mode.channels)]

    @property
    def height(self) -> int:
        return self._pil.height

    @property
    def width(self) -> int:
        return self._pil.width

    @property
    def resolution(self) -> tuple[int, int]:
        """Returns the (height, width) of the image."""
        return (self.height, self.width)

    @property
    def size(self) -> tuple[int, int, int, int]:
        """Returns the (batch, channels, height, width) of the image."""
        return (1, self._mode.channels, self.height, self.width)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Returns the (batch, channels, height, width) of the image."""
        return self.size

    def add_alpha(
        self,
        *,
        threshold: int = 0,
        color: Color = (0, 0, 0),
        constant_alpha: int | None = None,
        gradient_alpha: bool = False,
    ) -> "Image[RGBA]":
        """Add an alpha channel to the image.

        The alpha channel is computed by thresholding the image at the
        given threshold. The alpha channel is constant if constant_alpha
        is not None. If gradient_alpha is True, the alpha channel is
        computed by taking the gradient of the image.
        """
        image = self.convert(RGB)

        if constant_alpha is not None:
            alpha = Image.create_rectangle(
                image.resolution,
                color=(constant_alpha, constant_alpha, constant_alpha),
                mode=Grayscale,
            )
        else:
            diff_image = Image.from_numpy(
                image.to_numpy()
                - Image.create_rectangle(
                    image.resolution, color=color, mode=RGB
                ).to_numpy()
            )
            if gradient_alpha:
                diff = diff_image.convert(Grayscale).to_numpy()
                diff = np.gradient(diff)
                alpha_np = np.linalg.norm(diff, axis=0)
                alpha_np = np.where(alpha_np > threshold, alpha_np, 0)
                alpha = Image.from_numpy(alpha_np, mode=Grayscale)
            else:
                alpha_np = diff_image.convert(Grayscale).to_numpy()
                alpha_np = np.where(alpha_np > threshold, 255, 0)
                alpha = Image.from_numpy(alpha_np, mode=Grayscale)

        channels = image.channels + [alpha]
        return Image.merge_channels(*channels, mode=RGBA)

    def blur(self, *, radius: int = 2) -> "Image[T]":
        def apply_blur(pil: PILImage.Image) -> PILImage.Image:
            return pil.filter(filter=ImageFilter.GaussianBlur(radius=radius))

        return self._bind(apply_blur)

    def crop(self, box: tuple[Point, Point], /) -> "Image[T]":
        """Crop the image to the given box defined by the top-left and bottom-
        right corners."""
        pil_box = self._box_to_pil_box(box)
        return self._bind(lambda pil: pil.crop(box=pil_box))

    def convert(self, mode: type[U], /) -> "Image[U]":
        return Image(_pil=self._pil.convert(mode=mode().name), _mode=mode())

    def composite(
        self, image: "Image[T]", /, *, mask: "Image[Grayscale] | Image[RGBA]"
    ) -> "Image[T]":
        """Composite this image with another image using the given mask.

        The image needs to have the same shape with self and the mask
        needs to have the same resolution with self.
        """
        assert (
            self.shape == image.shape
        ), f"image shapes {self.shape} and {image.shape} do not match"
        assert self.resolution == mask.resolution, (
            f"image resolution {self.resolution} and mask resolution"
            f" {mask.resolution} do not match"
        )
        return self._bind(
            lambda pil: PILImage.composite(image1=pil, image2=image.pil, mask=mask.pil)
        )

    def find_bbox(self, *, threshold: float = 0.0) -> Box:
        """Find the bounding box of the image.

        The threshold is used to determine the minimum value of the
        image that should be considered as part of the object. The
        bounding box is defined by the top-left and bottom-right
        corners.
        """

        array = np.array(object=self._pil, dtype=np.float32) / 255.0
        y_idxs, x_idxs = np.where(array > threshold)

        if not y_idxs.any() or not x_idxs.any():
            return ((0, 0), (0, 0))

        return ((y_idxs.min(), x_idxs.min()), (y_idxs.max(), x_idxs.max()))

    def copy(self) -> "Image[T]":
        return self._bind(lambda pil: pil.copy())

    def invert(self, *, invert_alpha: bool = False) -> "Image[T]":
        """Invert the image.

        If the image is RGBA, only the RGB channels are inverted by
        default. Set invert_alpha to True to invert the alpha channel as
        well.
        """
        if self._mode.channels <= 3 or invert_alpha:
            return self._bind(ImageOps.invert)
        else:
            rgb_image = self.convert(RGB).invert()
            channels = rgb_image.channels + [self[-1]]
            return Image.merge_channels(*channels, mode=RGBA)  # type: ignore[reportGeneralTypeIssues]

    def paste(self, image: "Image[T]", /, *, box: Box) -> "Image[T]":
        """Paste the given image into this image at the given box defined by
        the top-left and bottom-right corners."""
        pil_box = self._box_to_pil_box(box)
        new_pil = self._pil.copy()
        new_pil.paste(im=image.pil, box=pil_box)
        return Image.from_pil(new_pil, mode=self.mode)

    def resize(self, size: tuple[int, int], /) -> "Image[T]":
        return self._bind(lambda pil: pil.resize(size=size))

    def rotate(self, angle: int, /, *, center: Point | None = None) -> "Image[T]":
        return self._bind(lambda pil: pil.rotate(angle=angle, center=center))

    def save(self, path: str | Path, /) -> None:
        self._pil.save(fp=path)

    def scale(self, factor: float) -> "Image[T]":
        """Scale the image by the given factor."""
        new_width = int(self.width * factor)
        new_height = int(self.height * factor)
        return self.resize((new_width, new_height))

    def fit(self, size: int, /) -> "Image[T]":
        """Resize the image by its larger dimension to the given size."""
        if self.width > self.height:
            new_width = size
            new_height = int(size * self.height / self.width)
        else:
            new_width = int(size * self.width / self.height)
            new_height = size
        return self.resize((new_width, new_height))

    def expand(self, size: int, /) -> "Image[T]":
        """Resize the image by its smaller dimension to the given size."""
        if self.width < self.height:
            new_width = size
            new_height = int(size * self.height / self.width)
        else:
            new_width = int(size * self.width / self.height)
            new_height = size
        return self.resize((new_width, new_height))

    def to_numpy(self) -> NDArray[np.float32]:
        return np.array(object=self._pil).transpose(2, 0, 1) / 255.0

    def to_tensor(self, device: str | Device, dtype: DType, /) -> Tensor:
        return torch.tensor(data=self.to_numpy(), device=device, dtype=dtype)  # type: ignore[reportUnknownMemberType]

    @staticmethod
    def create_rectangle(
        size: tuple[int, int],
        /,
        *,
        color: tuple[int, int, int] = (0, 0, 0),
        mode: type[U] = RGB,
    ) -> "Image[U]":
        pil = PILImage.new(mode=mode.name, size=size, color=color)
        return Image.from_pil(pil, mode=mode)

    @staticmethod
    def from_bytes(bytes: bytes, /, *, mode: type[U] = RGB) -> "Image[U]":
        return Image.from_pil(PILImage.open(fp=bytes), mode=mode)

    @staticmethod
    def from_numpy(array: ArrayLike, /, mode: type[U] = RGB) -> "Image[U]":
        return Image.from_pil(PILImage.fromarray(array=array), mode=mode)  # type: ignore[reportUnknownMemberType]

    @staticmethod
    def from_pil(pil: PILImage.Image, /, mode: type[U] = RGB) -> "Image[U]":
        return Image(_pil=pil, _mode=mode())

    @staticmethod
    def from_tensor(tensor: Tensor, /, *, mode: type[U] = RGB) -> "Image[U]":
        assert (
            tensor.shape[1] == mode.channels
        ), f"tensor shape {tensor.shape} does not match mode {mode}"
        tensor = (
            255 * tensor.detach().cpu().to(dtype=torch.float32).clamp(min=0, max=1)
        ).byte()
        np_array = tensor.squeeze(dim=0).permute(1, 2, 0).numpy()
        return Image.from_numpy(np_array, mode=mode)

    @staticmethod
    def merge_channels(
        *channels: "Image[Grayscale]", mode: type[U] = RGB
    ) -> "Image[U]":
        pil = PILImage.merge(
            mode=mode.name, bands=[channel.pil for channel in channels]
        )
        return Image.from_pil(pil, mode=mode)

    @staticmethod
    def open(path: str, /, *, mode: type[U] = RGB) -> "Image[U]":
        return Image.from_pil(PILImage.open(fp=path), mode=mode)

    def _bind(self, func: Callable[[PILImage.Image], PILImage.Image], /):
        return Image.from_pil(func(self._pil), mode=self._mode.__class__)

    def _box_to_pil_box(self, box: Box, /) -> tuple[int, int, int, int]:
        """Convert the given box defined by the top-left and bottom-right
        corners to a PIL box (left, upper, right, lower)."""
        return (box[0][1], box[0][0], box[1][1], box[1][0])

    def _post_init(self) -> None:
        assert self._mode in MODES, f"mode {self._mode} not in {MODES}"
        assert (
            self._pil.mode == self._mode.name
        ), f"image mode {self._pil.mode} does not match {self._mode.name}"
