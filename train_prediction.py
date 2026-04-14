"""Open-loop trajectory prediction — LightningCLI entry point."""

from lightning.pytorch.cli import LightningCLI

from anon_tokyo.data.datamodule import WOMDDataModule
from anon_tokyo.prediction.lit_module import PredictionModule


def main() -> None:
    LightningCLI(PredictionModule, WOMDDataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
