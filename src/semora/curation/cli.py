"""JSON-file CLI for trusted operators; no corpus writes."""

import argparse
import json
import sys
from pathlib import Path

from .store import CurationStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("operation", choices=("init", "list", "group", "mutate"))
    parser.add_argument("--request", type=Path, help="UTF-8 JSON operation envelope")
    parser.add_argument("--group-id")
    args = parser.parse_args()
    with CurationStore(
        args.store,
        corpus_id=args.corpus_id,
        initialize=args.operation == "init",
        read_only=args.operation in ("list", "group"),
    ) as store:
        if args.operation == "mutate":
            if args.request is None:
                parser.error("mutate requires --request")
            request = json.loads(args.request.read_text(encoding="utf-8"))
            result = store.mutate(**request)
        elif args.operation == "group":
            result = store.group(args.group_id)
        elif args.operation == "list":
            result = {"revision": store.metadata()["revision"], "groups": store.groups()}
        else:
            result = store.metadata()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
