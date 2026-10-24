"""Keep only a completed Cube arm's final model after its evaluations exist."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()

    assert args.arm and all(char.isalnum() or char in "-_" for char in args.arm)
    name = f"cube-{args.arm}-seed3072"
    run_root = args.run_root.resolve()
    checkpoint_dir = (run_root / "artifacts" / "checkpoints" / name).resolve()
    assert checkpoint_dir.parent == run_root / "artifacts" / "checkpoints"
    final = checkpoint_dir / "weights_epoch_20.pt"
    evaluations = [
        run_root / "evaluation" / name / f"epoch-{epoch}" / condition / "ogb_cube_results.txt"
        for epoch in (1, 20) for condition in ("clean", "gaussian")
    ]
    assert final.is_file() and final.stat().st_size > 100_000_000
    assert all(path.is_file() and path.stat().st_size > 0 for path in evaluations)

    intermediate = [
        path for path in checkpoint_dir.glob("weights_epoch_*.pt")
        if path.name != final.name
    ]
    intermediate.extend((checkpoint_dir / "training_state").glob("*.ckpt"))
    assert all(
        path.is_file() and path.parent in {checkpoint_dir, checkpoint_dir / "training_state"}
        for path in intermediate
    )
    freed = sum(path.stat().st_size for path in intermediate)
    for path in intermediate:
        path.unlink()
    print(
        f"checkpoint_cleanup arm={args.arm} removed={len(intermediate)} "
        f"freed_gib={freed / 2**30:.3f} retained={final}",
        flush=True,
    )


if __name__ == "__main__":
    main()
