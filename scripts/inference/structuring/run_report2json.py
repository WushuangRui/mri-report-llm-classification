"""
Script name:
    raw_report_to_json.py

Purpose:
    This script reads a CSV file containing raw MRI reports, sends each report
    to a local vLLM model with a predefined prompt, and writes the generated
    JSON-style outputs to a new CSV file.

Input:
    A CSV file containing at least:
    - one report ID column named "id"
    - one raw report text column named "raw_report"

Output:
    A CSV file containing:
    - id
    - json_report

Notes:
    - Update the model path, input CSV path, and output CSV path below before use.
    - The prompt file in GitHub is expected at:
      prompts/prompt_raw2json.txt
"""

import csv
from pathlib import Path

from tqdm import tqdm
from vllm import LLM
from vllm.utils import FlexibleArgumentParser


# ============================================================
# Paths to modify
# ============================================================
MODEL_PATH = "path/to/your/local/model"
INPUT_CSV_PATH = Path("path/to/your/input_reports.csv")
OUTPUT_CSV_PATH = Path("path/to/your/output_results.csv")
PROMPT_FILE_PATH = Path("prompts/prompt_raw2json.txt")


# ============================================================
# Model runtime configuration
# ============================================================
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_BATCH_SIZE = 8


def load_prompt_text(prompt_file_path: Path) -> str:
    """
    Load the raw-to-JSON prompt text from a file.

    Args:
        prompt_file_path: Path to the prompt text file.

    Returns:
        The prompt content as a stripped string.
    """
    with open(prompt_file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def build_chat_messages(prompt_text: str, report_id: str, report_text: str) -> list[dict]:
    """
    Build the chat message sequence for one report.

    The original logic is preserved:
    - a generic system message
    - the prompt text as a user message
    - an example input/output pair
    - the current report as the final user message

    Args:
        prompt_text: Prompt instructions loaded from file.
        report_id: Report ID.
        report_text: Raw report text.

    Returns:
        A list of chat message dictionaries compatible with vLLM chat input.
    """
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt_text},
        {"role": "user", "content": "Here's an example of how to process a report:"},
        {
            "role": "user",
            "content": """
Example Input:
ID: 6
input: 
Clinical indication:
39-year-old female headache.

Technique:
Multiplanar multi-sequence MR images of the brain were obtained without intravenous contrast administration.

Comparison:
Brain MRI 3/21/2013

Findings:
The left mesial precentral gyrus cavernoma shows ill-defined T2/FLAIR hyperintense and T1-isotense signal extending inferolaterally, which may represent edema. There is mild interval increase in size of the lesion, currently measuring 1.5 x 1.3 cm, previously 1.3 x 1.0 cm. The lesion remains heterogenous centrally, demonstrating T1 and T2 hyperintensity , a T1-isointense/T2-hypointense rim, and susceptibility. Mild associated gyral expansion is similar to prior MRI. There is no midline shift. These findings are most compatible cavernoma which has subacute hemorrhage with new mild edema in the adjacent parenchyma. Linear T1 isointense focus just lateral to the cavernoma is compatible with a developmental venous anomaly, seen on the postcontrast images of the prior exam. The ventricular system is of normal size and configuration. There is no midline shift or herniation pattern. No extra axial collections are identified. Normal flow voids of the major intracranial vessels are seen on T2 weighted spin echo imaging. There is no evidence of a diffusion abnormality to suggest acute or subacute infarction. Small right maxillary retention cyst. The paranasal sinuses, mastoid air cells and orbits are unremarkable.

Impression:
Mild interval increase in size with associated minimal vasogenic edema of the paramedian left precentral gyral cavernous malformation.

Others:
Final Report: Dictated by Resident Lindsay Griffin MD and Signed by Attending Mari Hagiwara MD 11/12/2014 11:31 AM

Example Output:
{
  "quality_dec": false,
  "clinical_indication": [
    {
      "age": 39,
      "gender": "female",
      "symptom": ["headache"],
      "history": null
    }
  ],
  "comparison": [
    {
      "date_mri": "11/12/2014",
      "date_pre": "3/21/2013",
      "modality": "Brain MRI"
    }
  ],
  "lesions": [
    {
      "type": "cavernoma",
      "location": {
        "side": "left",
        "region": ["mesial precentral gyrus"]
      },
      "size": {
        "dimensions": "1.5 x 1.3",
        "units": "cm"
      },
      "radiological_changes": {
        "T2/FLAIR": ["hyperintense"],
        "T1": ["isointense"]
      },
      "characteristics": [
        "ill-defined",
        "heterogenous centrally",
        "T1-isointense/T2-hypointense rim",
        "susceptibility",
        "mild associated gyral expansion"
      ],
      "compare_pre": {
        "size of the lesion increase": "mild"
      },
      "diagnosis": {
        "cavernoma": ["compatible"]
      }
    },
    {
      "type": "developmental venous anomaly",
      "location": {
        "side": "left",
        "region": ["lateral to the cavernoma"]
      },
      "size": null,
      "radiological_changes": {
        "T1": ["isointense"]
      },
      "characteristics": ["linear"],
      "compare_pre": null
    }
  ],
  "diagnosis": {
    "developmental venous anomaly": ["compatible"]
  },
  "additional_findings": [
    {
      "finding": "small retention cyst",
      "location": "right maxillary"
    }
  ]
}
""",
        },
        {
            "role": "user",
            "content": f"Now, please process this report:\n\nID: {report_id}\ninput: {report_text}",
        },
    ]


def validate_input_rows(rows: list[dict]) -> None:
    """
    Run a lightweight validation check on input rows.

    The original behavior is preserved:
    - warn if the "raw_report" field appears to contain an unescaped newline
    - warn if the "raw_report" field is missing

    Args:
        rows: List of CSV rows loaded as dictionaries.
    """
    for row_index, row in enumerate(rows, start=1):
        try:
            if "\n" in row["raw_report"] and not row["raw_report"].startswith('"'):
                print(f"Warning: possible unescaped newline character at row {row_index}")
        except KeyError:
            print(f"Error: row {row_index} is missing the 'raw_report' field")


def load_reports_from_csv(input_csv_path: Path) -> list[dict]:
    """
    Load raw reports from the input CSV file.

    Args:
        input_csv_path: Path to the input CSV file.

    Returns:
        A list of row dictionaries.
    """
    with open(input_csv_path, "r", newline="", encoding="utf-8") as file:
        csv_reader = csv.DictReader(
            file,
            delimiter=",",
            quotechar='"',
            escapechar="\\",
        )
        rows = list(csv_reader)

    return rows


def initialize_llm(model_path: str) -> LLM:
    """
    Initialize the local vLLM model.

    Args:
        model_path: Local model path to load.

    Returns:
        An initialized vLLM LLM instance.
    """
    return LLM(
        model=model_path,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )


def configure_sampling_params(
    llm: LLM,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
):
    """
    Create and update sampling parameters for generation.

    Args:
        llm: Initialized vLLM model instance.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        top_k: Top-k sampling parameter.

    Returns:
        A sampling parameters object configured for generation.
    """
    sampling_params = llm.get_default_sampling_params()

    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    return sampling_params


def batch_process_reports(
    llm: LLM,
    sampling_params,
    prompt_text: str,
    reports: list[dict],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """
    Process reports in batches through the LLM chat interface.

    The original logic is preserved:
    - prompts are built report by report
    - batch inference is executed with llm.chat(...)
    - if one batch fails, all rows in that batch receive an error message

    Args:
        llm: Initialized vLLM model instance.
        sampling_params: Sampling parameters object.
        prompt_text: Prompt instructions loaded from file.
        reports: List of input report rows.
        batch_size: Number of reports to process per batch.

    Returns:
        A list of dictionaries with keys:
        - id
        - json_report
    """
    results = []

    for start_index in tqdm(range(0, len(reports), batch_size), desc="Processing batches"):
        batch_rows = reports[start_index:start_index + batch_size]

        batch_prompts = [
            build_chat_messages(prompt_text, row["id"], row["raw_report"])
            for row in batch_rows
        ]

        try:
            outputs = llm.chat(batch_prompts, sampling_params, use_tqdm=False)

            for batch_offset, output in enumerate(outputs):
                generated_text = output.outputs[0].text.strip()
                results.append(
                    {
                        "id": batch_rows[batch_offset]["id"],
                        "json_report": generated_text,
                    }
                )

        except Exception as error:
            print(f"[ERROR] Batch starting at index {start_index} failed: {error}")
            for row in batch_rows:
                results.append(
                    {
                        "id": row["id"],
                        "json_report": f"[ERROR] {str(error)}",
                    }
                )

    return results


def save_results_to_csv(results: list[dict], output_csv_path: Path) -> None:
    """
    Save processed results to a CSV file.

    Args:
        results: List of output dictionaries.
        output_csv_path: Path to the output CSV file.
    """
    with open(output_csv_path, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["id", "json_report"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main(parsed_args: dict) -> None:
    """
    Run the complete raw-report-to-JSON pipeline.

    Args:
        parsed_args: Parsed command-line arguments as a dictionary.
    """
    max_tokens = parsed_args.pop("max_tokens")
    temperature = parsed_args.pop("temperature")
    top_p = parsed_args.pop("top_p")
    top_k = parsed_args.pop("top_k")
    parsed_args.pop("chat_template_path", None)

    llm = initialize_llm(MODEL_PATH)
    sampling_params = configure_sampling_params(
        llm=llm,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    reports = load_reports_from_csv(INPUT_CSV_PATH)
    validate_input_rows(reports)

    prompt_text = load_prompt_text(PROMPT_FILE_PATH)

    results = batch_process_reports(
        llm=llm,
        sampling_params=sampling_params,
        prompt_text=prompt_text,
        reports=reports,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    save_results_to_csv(results, OUTPUT_CSV_PATH)

    print(f"Finished processing {len(results)} reports.")
    print(f"Results saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()

    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int, default=2048)
    sampling_group.add_argument("--temperature", type=float, default=0.7)
    sampling_group.add_argument("--top-p", type=float, default=0.95)
    sampling_group.add_argument("--top-k", type=int, default=40)

    parser.add_argument("--chat-template-path", type=str)

    args = vars(parser.parse_args())
    main(args)
