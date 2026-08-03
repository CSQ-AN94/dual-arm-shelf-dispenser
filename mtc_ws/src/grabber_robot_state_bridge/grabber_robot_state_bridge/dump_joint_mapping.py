#!/usr/bin/env python3
"""Write joint_state_mapping.json.  No ROS, no SDK, no robot required."""

import argparse
import json

from . import joint_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="joint_state_mapping.json")
    args = parser.parse_args()
    joint_map.validate()
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(joint_map.mapping_document(), handle, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
