#!/usr/bin/env python3
import argparse
import gzip
import io
import json
import random
import tarfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path


DELICIOUS_3D_URL = "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/delicious/delicious-3d.tns.gz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a 3D sparse .tns tensor for BTC-GPU by creating .tns, .train, and .test files."
    )
    parser.add_argument("--input", help="Input 3D .tns or .tns.gz file.")
    parser.add_argument("--dataset", required=True, help="Output dataset basename, e.g. delicious-3d.")
    parser.add_argument("--output-dir", default="/fan/data", help="Directory for output files.")
    parser.add_argument("--train-rate", type=float, default=0.8, help="Random train split rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--download", action="store_true", help="Download input if it does not exist.")
    parser.add_argument("--url", default=None, help="Download URL. Defaults to Delicious 3D URL.")
    parser.add_argument("--write-tns", action="store_true", help="Also write an uncompressed full .tns copy.")
    parser.add_argument(
        "--normalize",
        choices=("max", "none"),
        default="max",
        help="Normalize values. 'max' writes value / max_abs_value; 'none' keeps original values.",
    )
    parser.add_argument("--progress", type=int, default=5_000_000, help="Print progress every N rows; 0 disables.")
    return parser.parse_args()


def input_path_for(args):
    if args.input:
        return Path(args.input)
    return Path(args.output_dir) / f"{args.dataset}.tns.gz"


def download_if_needed(path, url):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


@contextmanager
def open_text_maybe_gzip(path):
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


def show_progress(label, rows, step):
    if step > 0 and rows % step == 0:
        print(f"{label}: processed {rows:,} rows")


def parse_entry(line):
    parts = line.split()
    if len(parts) < 4:
        return None

    try:
        return int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
    except ValueError:
        return None


def scan_3d_tns(input_path, progress):
    rows = 0
    skipped_rows = 0
    min_coords = None
    max_coords = None
    max_abs_value = 0.0

    with open_text_maybe_gzip(input_path) as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line or line[0] in "#%":
                continue

            entry = parse_entry(line)
            if entry is None:
                skipped_rows += 1
                continue

            x, y, z, value = entry
            coords = [x, y, z]
            if min_coords is None:
                min_coords = coords[:]
                max_coords = coords[:]
            else:
                for axis in range(3):
                    min_coords[axis] = min(min_coords[axis], coords[axis])
                    max_coords[axis] = max(max_coords[axis], coords[axis])

            max_abs_value = max(max_abs_value, abs(value))
            rows += 1
            show_progress("scan", rows, progress)

    return rows, skipped_rows, min_coords, max_coords, max_abs_value


def split_3d_tns(args):
    if not 0 < args.train_rate < 1:
        raise SystemExit("--train-rate must be between 0 and 1.")

    input_path = input_path_for(args)
    url = args.url or DELICIOUS_3D_URL
    if args.download:
        download_if_needed(input_path, url)
    if not input_path.exists():
        raise SystemExit(f"{input_path} does not exist. Provide --input or use --download.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tns_path = output_dir / f"{args.dataset}.tns"
    train_path = output_dir / f"{args.dataset}.train"
    test_path = output_dir / f"{args.dataset}.test"
    meta_path = output_dir / f"{args.dataset}.split.json"

    print(f"Input: {input_path}")
    print("Scanning input for dimensions and normalization scale...")
    rows, skipped_rows, min_coords, max_coords, max_abs_value = scan_3d_tns(input_path, args.progress)
    value_scale = max_abs_value if args.normalize == "max" and max_abs_value > 0 else 1.0

    rng = random.Random(args.seed)
    written_rows = 0
    train_rows = 0
    test_rows = 0

    print(f"Rows found: {rows:,}; skipped rows: {skipped_rows:,}")
    if args.normalize == "max":
        print(f"Normalizing values by max_abs_value={value_scale:.12g}")
    else:
        print("Normalization disabled.")
    if args.write_tns:
        print(f"Writing: {tns_path}")
    else:
        print("Skipping full .tns copy; use --write-tns if you need it.")
    print(f"Writing: {train_path}")
    print(f"Writing: {test_path}")

    with open_text_maybe_gzip(input_path) as source, \
            train_path.open("w", encoding="utf-8", newline="\n") as train_out, \
            test_path.open("w", encoding="utf-8", newline="\n") as test_out:
        tns_out = tns_path.open("w", encoding="utf-8", newline="\n") if args.write_tns else None

        try:
            for raw_line in source:
                line = raw_line.strip()
                if not line or line[0] in "#%":
                    continue

                entry = parse_entry(line)
                if entry is None:
                    continue

                x, y, z, value = entry
                out_value = value / value_scale
                out_line = f"{x} {y} {z} {out_value:.12g}\n"

                if tns_out is not None:
                    tns_out.write(out_line)
                if rng.random() < args.train_rate:
                    train_out.write(out_line)
                    train_rows += 1
                else:
                    test_out.write(out_line)
                    test_rows += 1

                written_rows += 1
                show_progress("write", written_rows, args.progress)
        finally:
            if tns_out is not None:
                tns_out.close()

    coord_base = 0 if min_coords and min(min_coords) == 0 else 1
    metadata = {
        "dataset": args.dataset,
        "input": str(input_path),
        "outputs": {
            "tns": str(tns_path) if args.write_tns else None,
            "train": str(train_path),
            "test": str(test_path),
        },
        "train_rate": args.train_rate,
        "test_rate": 1.0 - args.train_rate,
        "seed": args.seed,
        "rows": rows,
        "skipped_rows": skipped_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "min_coords": min_coords,
        "max_coords": max_coords,
        "coordinate_base": coord_base,
        "normalization": args.normalize,
        "value_scale": value_scale,
        "max_abs_value": max_abs_value,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Rows: {rows:,}; train: {train_rows:,}; test: {test_rows:,}")
    print(f"Normalization: {args.normalize}; value_scale: {value_scale:.12g}")
    print(f"Coordinate base detected: {coord_base}")
    print(f"Metadata: {meta_path}")


def main():
    args = parse_args()
    split_3d_tns(args)


if __name__ == "__main__":
    main()
