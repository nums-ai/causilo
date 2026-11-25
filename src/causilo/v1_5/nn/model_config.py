"""Architecture and checkpoint dimensions for Causilo 1.5."""

from dataclasses import asdict, dataclass

ROW_LATENTS = 4
NATIVE_CLASSES = 10
NUM_QUANTILES = 999
FOURIER_FREQUENCIES = 16
ECDF_FREQUENCIES = 4
EXPANSION = 2
COLUMN_HEADS = 4
COLUMN_LATENTS = 256
ROW_HEADS = 8
PREDICTION_HEADS = 8


@dataclass(frozen=True)
class ModelConfig:
    """Widths and depths of the paired column/row stages and prediction block."""

    max_classes: int = NATIVE_CLASSES
    width: int = 256
    column_depth: int = 8
    prediction_depth: int = 24

    def __post_init__(self) -> None:
        if type(self.max_classes) is not int or self.max_classes not in (0, NATIVE_CLASSES):
            raise ValueError(f"max_classes must be 0 for regression or {NATIVE_CLASSES} for classification")
        for name, value in self.to_dict().items():
            if name != "max_classes" and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.width % ROW_HEADS:
            raise ValueError(f"width must be divisible by {ROW_HEADS}")

    @property
    def prediction_width(self) -> int:
        return self.width * ROW_LATENTS

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
