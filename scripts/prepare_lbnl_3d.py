#!/usr/bin/env python3
import argparse
import gzip
import io
import json
import random
import tarfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


LBNL_5D_DIMS = (1605, 4198, 1631, 4209, 868131)
DEFAULT_KEEP_MODES = (1, 3, 5)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert FROSTT LBNL 5D network tensor to a 3D sparse tensor. "
            "Default 3D modes are sender_ip, destination_ip, time; "
            "sender_port and destination_port are summed out."
        )
    )
    parser.add_argument("--input", required=True, help="Input lbnl-network .tns, .tns.gz, or tar.gz path.")
    parser.add_argument("--dataset", default="lbnl-network-3d", help="Output dataset basename.")
    parser.add_argument("--output-dir", default="/hy-tmp/fan/data", help="Directory for output files.")
    parser.add_argument(
        "--keep-modes",
        default="1,3,5",
        help="Three 1-based modes to keep, e.g. 1,3,5 means sender_ip,destination_ip,time.",
    )
    parser.add_argument("--train-rate", type=float, default=0.8, help="Train split rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--write-tns", action="store_true", help="Also write the converted full .tns file.")
    parser.add_argument(
        "--normalize",
        choices=("max", "none"),
        default="max",
        help="Normalize aggregated values. 'max' writes value / max_abs_value; 'none' keeps original values.",
    )
    parser.add_argument("--progress", type=int, default=1_000_000, help="Print progress every N rows; 0 disables.")
    return parser.parse_args()


@contextmanager
def open_text_tensor(path):
    path = Path(path)
    if str(path).endswith((".gz", ".tgz")):
        try:
            with tarfile.open(path, "r:*") as archive:
                members = [
                    member for member in archive.getmembers()
                    if member.isfile()
                    and not Path(member.name).name.startswith("._")
                    and member.name.lower().endswith(".tns")
                ]
                if members:
                    member = members[0]
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise SystemExit(f"Could not extract {member.name} from {path}.")
                    print(f"Reading tar member: {member.name}")
                    with io.TextIOWrapper(extracted, encoding="utf-8", errors="replace") as handle:
                        yield handle
                    return
        except tarfile.TarError:
            pass

    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield handle
        return

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        yield handle


def parse_keep_modes(text):
    modes = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if len(modes) != 3 or any(mode < 1 or mode > 5 for mode in modes):
        raise SystemExit("--keep-modes must contain exactly three 1-based modes in [1,5], e.g. 1,3,5.")
    if len(set(modes)) != 3:
        raise SystemExit("--keep-modes cannot contain duplicate modes.")
    return modes


def show_progress(label, rows, step):
    if step > 0 and rows % step == 0:
        print(f"{label}: {rows:,} rows")


def aggregate_to_3d(input_path, keep_modes, progress):
    keep_indices = [mode - 1 for mode in keep_modes]
    values = defaultdict(float)
    rows = 0
    skipped = 0

    with open_text_tensor(input_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line[0] in "#%":
                continue

            parts = line.split()
            if len(parts) < 6:
                skipped += 1
                continue

            try:
                coords5 = [int(parts[i]) for i in range(5)]
                value = float(parts[5])
            except ValueError:
                skipped += 1
                continue

            key = tuple(coords5[idx] for idx in keep_indices)
            values[key] += value
            rows += 1
            show_progress("aggregate", rows, progress)

    return values, rows, skipped


def split_and_write(values, args, keep_modes):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tns_path = output_dir / f"{args.dataset}.tns"
    train_path = output_dir / f"{args.dataset}.train"
    test_path = output_dir / f"{args.dataset}.test"
    meta_path = output_dir / f"{args.dataset}.split.json"

    rng = random.Random(args.seed)
    train_rows = 0
    test_rows = 0
    min_coords = None
    max_coords = None
    max_abs_value = 0.0
    for value in values.values():
        max_abs_value = max(max_abs_value, abs(value))
    value_scale = max_abs_value if args.normalize == "max" and max_abs_value > 0 else 1.0

    if args.write_tns:
        print(f"Writing: {tns_path}")
    else:
        print("Skipping full converted .tns copy; use --write-tns if you need it.")
    if args.normalize == "max":
        print(f"Normalizing values by max_abs_value={value_scale:.12g}")
    else:
        print("Normalization disabled.")
    print(f"Writing: {train_path}")
    print(f"Writing: {test_path}")

    sorted_items = sorted(values.items())
    with train_path.open("w", encoding="utf-8", newline="\n") as train_out, \
            test_path.open("w", encoding="utf-8", newline="\n") as test_out:
        tns_out = tns_path.open("w", encoding="utf-8", newline="\n") if args.write_tns else None
        try:
            for coords, value in sorted_items:
                if value == 0:
                    continue
                out_value = value / value_scale
                out_line = f"{coords[0]} {coords[1]} {coords[2]} {out_value:.12g}\n"

                if min_coords is None:
                    min_coords = list(coords)
                    max_coords = list(coords)
                else:
                    for axis in range(3):
                        min_coords[axis] = min(min_coords[axis], coords[axis])
                        max_coords[axis] = max(max_coords[axis], coords[axis])

                if tns_out is not None:
                    tns_out.write(out_line)
                if rng.random() < args.train_rate:
                    train_out.write(out_line)
                    train_rows += 1
                else:
                    test_out.write(out_line)
                    test_rows += 1
        finally:
            if tns_out is not None:
                tns_out.close()

    coord_base = 0 if min_coords and min(min_coords) == 0 else 1
    output_dims = [LBNL_5D_DIMS[mode - 1] for mode in keep_modes]
    metadata = {
        "input": args.input,
        "dataset": args.dataset,
        "source_tensor": "FROSTT LBNL-Network",
        "original_modes": ["sender_ip", "sender_port", "destination_ip", "destination_port", "time"],
        "original_dimensions": LBNL_5D_DIMS,
        "kept_modes_1_based": keep_modes,
        "kept_mode_names": [["sender_ip", "sender_port", "destination_ip", "destination_port", "time"][m - 1] for m in keep_modes],
        "aggregation": "sum over dropped modes",
        "output_dimensions": output_dims,
        "aggregated_nnz": train_rows + test_rows,
        "train_rate": args.train_rate,
        "test_rate": 1.0 - args.train_rate,
        "seed": args.seed,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "min_coords": min_coords,
        "max_coords": max_coords,
        "coordinate_base": coord_base,
        "normalization": args.normalize,
        "value_scale": value_scale,
        "max_abs_value": max_abs_value,
        "outputs": {
            "tns": str(tns_path) if args.write_tns else None,
            "train": str(train_path),
            "test": str(test_path),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Aggregated nnz: {train_rows + test_rows:,}; train: {train_rows:,}; test: {test_rows:,}")
    print(f"Output dimensions: {output_dims[0]} {output_dims[1]} {output_dims[2]}")
    print(f"Normalization: {args.normalize}; value_scale: {value_scale:.12g}")
    print(f"Coordinate base detected: {coord_base}")
    print(f"Metadata: {meta_path}")


def main():
    args = parse_args()
    if not 0 < args.train_rate < 1:
        raise SystemExit("--train-rate must be between 0 and 1.")

    keep_modes = parse_keep_modes(args.keep_modes)
    print(f"Input: {args.input}")
    print(f"Keeping modes: {keep_modes}")
    print("Aggregating to 3D...")
    values, rows, skipped = aggregate_to_3d(args.input, keep_modes, args.progress)
    print(f"Read rows: {rows:,}; skipped rows: {skipped:,}; aggregated keys: {len(values):,}")
    split_and_write(values, args, keep_modes)


if __name__ == "__main__":
    main()
