"""
One-off helper: inspect a .pt checkpoint to figure out what's actually inside
it, so dental_qc/model2_rct.py's build_architecture() (or the loading logic
entirely) can be written to match.

Usage:
    python inspect_checkpoint.py models/best_RCT.pt
"""

import sys

import torch


def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_checkpoint.py <path-to-checkpoint.pt>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading {path} ...\n")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    print("=== Top level ===")
    print(f"type: {type(checkpoint)}")

    if isinstance(checkpoint, torch.nn.Module):
        print("\nThis is a full pickled model (nn.Module) -- model2_rct.py's")
        print("default loading path (torch.save(model, path)) already handles this.")
        print(f"Class: {checkpoint.__class__.__module__}.{checkpoint.__class__.__name__}")
        return

    if not isinstance(checkpoint, dict):
        print(f"\nUnrecognized checkpoint type: {type(checkpoint)}")
        return

    print(f"\nTop-level keys: {list(checkpoint.keys())}")

    # Ultralytics-style checkpoints usually have a 'model' (and often 'ema') key
    # holding the real pickled model object -- a strong sign to use
    # MODEL2_BACKEND = "yolo_classify" instead of the plain-classifier path.
    for key in ("model", "ema"):
        if key in checkpoint and isinstance(checkpoint[key], torch.nn.Module):
            val = checkpoint[key]
            print(f"\n'{key}' key found and is an nn.Module -- this looks like an")
            print("Ultralytics YOLO checkpoint. Try setting, in dental_qc/config.py:")
            print('    MODEL2_BACKEND = "yolo_classify"')
            print(f"  (class: {val.__class__.__module__}.{val.__class__.__name__})")
            return

    state_dict = checkpoint.get("state_dict", checkpoint)
    items = list(state_dict.items()) if hasattr(state_dict, "items") else None

    if items and hasattr(items[0][1], "shape"):
        print(f"\n=== Looks like a raw state_dict ({len(items)} parameters) ===")
        print("First 25 parameter names and shapes:")
        for name, tensor in items[:25]:
            print(f"  {name:50s} {tuple(tensor.shape)}")
        if len(items) > 25:
            print(f"  ... and {len(items) - 25} more")
        print("\nShare this output (or the architecture name, if you recognize it --")
        print("e.g. ResNet18, EfficientNet-B0, a custom CNN) and build_architecture()")
        print("in dental_qc/model2_rct.py can be filled in to match.")
    else:
        print("\nDidn't find a clear nn.Module or state_dict under the usual keys.")
        print("Paste the 'Top-level keys' output above and I can help from there.")


if __name__ == "__main__":
    main()
