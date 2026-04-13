import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lecture video -> IR schemas pipeline (thesis-final-pipeline)"
    )
    parser.add_argument("video", type=Path, help="Path to the input lecture video file")
    parser.add_argument("output_dir", type=Path, help="Directory to write pipeline outputs")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    from pipeline.pipeline import run
    run(args.video, args.output_dir)


if __name__ == "__main__":
    main()
