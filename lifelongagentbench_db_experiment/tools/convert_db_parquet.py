"""Convert the Hugging Face DB parquet into the JSON files LifelongAgentBench loads.

The published parquet stores nested fields as Python-literal strings, while
src/tasks/instance/db_bench/task.py expects a JSON dict keyed by sample index.
Also writes the flat file src/agents/instance/oracle_agent.py reads, which has a
different layout from the task's data file.

Usage:
    python convert_db_parquet.py <path to db_bench parquet> <path to LifelongAgentBench repo>
"""

import ast
import json
import os
import sys

import pyarrow.parquet as pq

OUTPUT_SUBDIR = "data/v0303/db_bench/processed/v0317_first500"


def parse(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def main(parquet_path: str, repo_dir: str) -> None:
    output_dir = os.path.join(repo_dir, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    row_list = pq.read_table(parquet_path).to_pylist()
    row_list.sort(key=lambda row: row["sample_index"])
    entry_dict = {}
    oracle_dict = {}
    for row in row_list:
        key = str(row["sample_index"])
        answer_info = parse(row["answer_info"])
        if answer_info["direct"] is not None:
            answer_info["direct"] = [list(item) for item in answer_info["direct"]]
        entry_dict[key] = {
            "answer_info": answer_info,
            "table_info": parse(row["table_info"]),
            "instruction": row["instruction"],
            "skill_list": parse(row["skill_list"]),
        }
        oracle_dict[key] = {
            "instruction": row["instruction"],
            "sql": " ".join(answer_info["sql"].split()),
            "answer_direct": answer_info["direct"],
            "answer_md5": answer_info["md5"],
        }
    for file_name, data in [
        ("entry_dict.json", entry_dict),
        ("oracle_dict.json", oracle_dict),
    ]:
        with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
    direct_count = sum(e["answer_info"]["direct"] is not None for e in entry_dict.values())
    print(f"entries: {len(entry_dict)} | direct: {direct_count} | md5: {len(entry_dict) - direct_count}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
