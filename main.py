from src.common.utils import load_config
import os

def main():
    # Load base config to check framework, or just load pytorch config if we know it's pytorch?
    # The instruction implies we check config.framework.
    # We should probably load the config that is relevant.
    # But how do we know which one to load?
    # Usually we load a default or look at args.
    # For this task, let's assume we load `configs/pytorch.yaml` if it exists, or `configs/base.yaml`.
    # But `pytorch.yaml` overrides `base.yaml`.
    # Let's try to load `configs/pytorch.yaml` directly as it should contain `framework: pytorch`.

    # However, if we want to be generic, we might need an argument.
    # But the instruction is simple.

    cfg = load_config("configs/pytorch.yaml")

    if cfg.get('framework') == "pytorch":
        from src.pytorch.train import main_pytorch
        main_pytorch()
    else:
        print(f"Framework {cfg.get('framework')} not implemented yet.")

if __name__ == "__main__":
    main()
