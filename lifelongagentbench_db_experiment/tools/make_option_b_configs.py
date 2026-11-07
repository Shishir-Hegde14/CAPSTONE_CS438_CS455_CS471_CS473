"""Write the three Option B experiment configs next to standard.yaml.

The patch already contains the generated configs; this script only documents how they were made.

Usage:
    python make_option_b_configs.py <path to LifelongAgentBench repo>
"""

import json
import os
import sys

CONFIG_SUBDIR = "configs/assignments/experiments/llama_31_8b_instruct_ollama/instance/db_bench/instance"


def order(start: int, stop: int) -> str:
    return json.dumps([str(i) for i in range(start, stop)])


def write(config_dir: str, source: str, target: str, start: int, stop: int, run: str, seed: bool) -> None:
    with open(os.path.join(config_dir, source), encoding="utf-8") as f:
        text = f.read()
    assert "output_dir: outputs/{TIMESTAMP}" in text and "sample_order: default" in text
    text = text.replace("output_dir: outputs/{TIMESTAMP}", f"output_dir: outputs/{run}")
    text = text.replace("sample_order: default", f"sample_order: {order(start, stop)}")
    if seed:
        line = "        utilized_sample_count: 4\n"
        assert line in text
        text = text.replace(
            line, line + "        seed_session_list_path: outputs/optb_run1/runs.json\n"
        )
    with open(os.path.join(config_dir, target), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    config_dir = os.path.join(sys.argv[1], CONFIG_SUBDIR)
    write(config_dir, "standard.yaml", "optb_run1.yaml", 0, 100, "optb_run1", False)
    write(config_dir, "standard.yaml", "optb_run2.yaml", 100, 200, "optb_run2", False)
    write(config_dir, "previous_sample_utilization_usc4.yaml", "optb_run3.yaml", 100, 200, "optb_run3", True)
