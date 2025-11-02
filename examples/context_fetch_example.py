"""Quick demo for ProjectContextFetcher.

Reads the first entry in the dataset, fetches source-context information from
local clones, and prints the key snippets.
"""
from pathlib import Path
import csv

from utils import ProjectContextFetcher

DATASET_PATH = Path("dataset/FlakyLens_dataset_with_nonflaky_indented.csv")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    with DATASET_PATH.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        first_row = next(reader)

    project = first_row["project"]
    test_name = first_row["test_name"]

    print(f"Project: {project}")
    print(f"Test: {test_name}\n")

    fetcher = ProjectContextFetcher()
    context = fetcher.get_test_context(
        project=project,
        test_name=test_name,
        context_lines=8,
        invocation_limit=5,
    )

    print(f"File: {context['file_path']}")
    print(f"Class: {context['class_name']}")
    print(f"Method: {context['method_name']}\n")

    annotations = context.get("annotations")
    if annotations:
        print("Annotations:\n" + annotations + "\n")

    print("Method block:\n" + context["method_block"] + "\n")
    print("Surrounding window:\n" + context["surrounding_window"] + "\n")

    if context["invocations"]:
        print("Call sites:")
        for idx, match in enumerate(context["invocations"], start=1):
            print(
                f"  [{idx}] {match['file_path']}:{match['line_number']} - "
                f"{match['line_preview']}"
            )
    else:
        print("No invocation sites found.")


if __name__ == "__main__":
    main()
