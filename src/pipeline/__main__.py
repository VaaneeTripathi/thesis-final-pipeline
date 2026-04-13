import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="..."
    )
    parser.add_argument("video", type=Path, help="Path to the input lecture video file")
    parser.add_argument("output_dir", type=Path, help="Directory to write pipeline outputs")
    args = parser.parse_args()

    # validate inputs exist
    # call pipeline.run(args.video, args.output_dir)
    # (pipeline.py doesn't exist yet — leave as a comment/stub)

if __name__ == "__main__":
    main()