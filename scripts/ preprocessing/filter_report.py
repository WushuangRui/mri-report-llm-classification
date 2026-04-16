"""
Script name:
    filter_spine_reports_and_no_impression.py

Purpose:
    This script reads a raw MRI report CSV file, automatically detects the ID
    column and report text column, and generates two filtered output files:

    1. Spine-related reports based on method-section rules, vertebral level
       mentions, and brain-related exclusion criteria.
    2. Reports that do not contain an "Impression" section.

Input:
    A CSV file containing at least:
    - one ID column (e.g., mid, id, study_id, exam_id)
    - one report text column (e.g., report, text, note, narrative, content)

Output:
    1. reports_condition1_spine_filtered.csv
       - spine-related reports with review fields
    2. reports_no_impression.csv
       - reports without an Impression section
"""

from pathlib import Path
import re
import pandas as pd


# ============================================================
# Paths to modify
# ============================================================
INPUT_CSV_PATH = Path("input_reports.csv")
OUTPUT_SPINE_FILTERED_PATH = Path("reports_condition1_spine_filtered.csv")
OUTPUT_NO_IMPRESSION_PATH = Path("reports_no_impression.csv")


# ============================================================
# Regular expressions and keyword definitions
# ============================================================

# Match method sections such as:
# Technique: ...
# Procedure: ...
# Exam: ...
# Capture content from the keyword to the first period.
METHOD_SECTION_PATTERN = re.compile(
    r"\b(technique|procedure|exam)\b\s*[:：]\s*(.*?)\.",
    re.IGNORECASE | re.DOTALL,
)

SPINE_PATTERN = re.compile(r"\bspine\b", re.IGNORECASE)
BRAIN_PATTERN = re.compile(r"\bbrain\b|\bbrian\b", re.IGNORECASE)  # includes common typo
HEAD_PATTERN = re.compile(r"\bhead\b", re.IGNORECASE)
IMPRESSION_PATTERN = re.compile(r"\bimpression\b", re.IGNORECASE)

VERTEBRAL_LEVEL_TOKENS = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5",
    "S1", "S2", "S3", "S4", "S5",
    "Co1", "Co2", "Co3", "Co4", "Co5",
]

VERTEBRAL_LEVEL_PATTERNS = {
    token: re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
    for token in VERTEBRAL_LEVEL_TOKENS
}

BRAIN_STRUCTURE_TERMS = [
    "cerebral hemispheres",
    "white matter",
    "grey matter",
    "gray matter",
    "basal ganglia",
    "thalamus",
    "brainstem",
    "brain stem",
    "cerebellum",
    "cerebellar",
    "pons",
    "medulla",
    "midbrain",
    "ventricles",
    "lateral ventricles",
    "third ventricle",
    "fourth ventricle",
    "corpus callosum",
    "pituitary",
    "sella",
    "hypothalamus",
    "hippocampus",
    "temporal lobe",
    "frontal lobe",
    "parietal lobe",
    "occipital lobe",
    "insula",
    "optic nerve",
    "optic nerves",
    "optic chiasm",
    "periventricular",
    "juxtacortical",
    "infratentorial",
    "supratentorial",
    "flair",
    "t2 flair",
    "t2/flair",
]

BRAIN_STRUCTURE_PATTERNS = [
    re.compile(re.escape(term), re.IGNORECASE)
    for term in BRAIN_STRUCTURE_TERMS
]

BRAIN_HEAD_MRI_MRA_LABEL_PATTERN = re.compile(
    r"\b("
    r"(brain|head)\s+(mri|mra)"
    r"|"
    r"(mri|mra)\s+(brain|head)"
    r")\b\s*[:：]",
    re.IGNORECASE,
)


# ============================================================
# Helper functions
# ============================================================

def detect_id_and_report_columns(dataframe: pd.DataFrame) -> tuple[str, str]:
    """
    Automatically detect the ID column and the report text column.

    Detection rules:
    - ID column: exact match among {"mid", "id", "study_id", "exam_id"} (case-insensitive)
    - Report column: column name contains one of
      {"report", "text", "note", "narrative", "content"} (case-insensitive)

    Fallback rules:
    - If no ID column is detected, use the first column.
    - If no report column is detected, select the object-type column with the
      largest average string length. If no object-type column exists, use the
      last column.

    Args:
        dataframe: Input DataFrame.

    Returns:
        A tuple of (id_column_name, report_column_name).
    """
    id_column = None
    report_column = None

    for column in dataframe.columns:
        column_lower = column.lower()

        if id_column is None and column_lower in {"mid", "id", "study_id", "exam_id"}:
            id_column = column

        if report_column is None and any(
            keyword in column_lower
            for keyword in ["report", "text", "note", "narrative", "content"]
        ):
            report_column = column

    if id_column is None:
        id_column = dataframe.columns[0]

    if report_column is None:
        candidate_columns = [
            (dataframe[column].astype(str).str.len().mean(), column)
            for column in dataframe.columns
            if dataframe[column].dtype == object
        ]
        report_column = max(candidate_columns, default=(None, dataframe.columns[-1]))[1]

    return id_column, report_column


def extract_method_sections(report_text: str) -> list[str]:
    """
    Extract all method sections that begin with 'Technique', 'Procedure', or 'Exam'
    and continue until the first period.

    Examples of supported section headers:
    - Technique:
    - Procedure:
    - Exam:

    Args:
        report_text: Full report text.

    Returns:
        A list of extracted method section strings. Returns an empty list if no
        matching section is found.
    """
    sections = []

    for match in METHOD_SECTION_PATTERN.finditer(report_text):
        section_text = match.group(2).strip()
        if section_text:
            sections.append(section_text)

    return sections


def count_unique_vertebral_levels(report_text: str) -> int:
    """
    Count how many distinct vertebral level tokens appear in the report.

    For example, if the report contains C3, C4, C5, T12, L4, and L5,
    the count is 6.

    Args:
        report_text: Full report text.

    Returns:
        Number of distinct vertebral level tokens found.
    """
    count = 0

    for _, pattern in VERTEBRAL_LEVEL_PATTERNS.items():
        if pattern.search(report_text):
            count += 1

    return count


def count_brain_structure_mentions(report_text: str) -> int:
    """
    Count how many distinct brain structure terms are mentioned in the report.

    This count is based on the predefined brain structure term list and does
    not count repeated occurrences of the same term multiple times.

    Args:
        report_text: Full report text.

    Returns:
        Number of distinct brain structure terms found.
    """
    count = 0

    for pattern in BRAIN_STRUCTURE_PATTERNS:
        if pattern.search(report_text):
            count += 1

    return count


def join_method_sections(method_sections: list[str]) -> str:
    """
    Join extracted method sections into a single review string.

    Args:
        method_sections: List of extracted method section strings.

    Returns:
        A single string joined by ' || '.
    """
    return " || ".join(method_sections)


def build_output_columns(
    output_dataframe: pd.DataFrame,
    base_columns: list[str],
    optional_review_columns: list[str],
) -> list[str]:
    """
    Build the final ordered output column list.

    Column ordering strategy:
    1. Keep base columns first (typically ID and report columns)
    2. Append optional review columns if they exist
    3. Append any remaining original columns except the internal helper column

    Args:
        output_dataframe: DataFrame to be exported.
        base_columns: Core columns to keep at the front.
        optional_review_columns: Review-related columns to append next.

    Returns:
        Ordered list of column names for CSV export.
    """
    ordered_columns = base_columns + [
        column for column in optional_review_columns
        if column in output_dataframe.columns
    ]

    ordered_columns += [
        column for column in output_dataframe.columns
        if column not in ordered_columns and column != "_report_text"
    ]

    return ordered_columns


def main() -> None:
    """
    Run the full filtering workflow.

    Workflow summary:
    1. Read the input CSV.
    2. Detect the ID column and report text column.
    3. Standardize the report text into an internal helper column.
    4. Compute method-section features, vertebral level counts, and brain-related flags.
    5. Generate:
       - condition 1 output: spine-related reports
       - condition 2 output: reports without an Impression section
    6. Save both output CSV files.
    7. Print a concise processing summary.
    """
    dataframe = pd.read_csv(INPUT_CSV_PATH)

    id_column, report_column = detect_id_and_report_columns(dataframe)

    dataframe["_report_text"] = dataframe[report_column].fillna("").astype(str)

    method_sections_series = dataframe["_report_text"].apply(extract_method_sections)
    unique_vertebral_level_counts = dataframe["_report_text"].apply(count_unique_vertebral_levels)
    brain_structure_counts = dataframe["_report_text"].apply(count_brain_structure_mentions)

    has_any_brain_structure = brain_structure_counts > 0
    has_too_many_brain_structures = brain_structure_counts > 5
    has_method_section = method_sections_series.apply(lambda sections: len(sections) > 0)

    any_method_section_contains_brain_or_head = method_sections_series.apply(
        lambda sections: any(
            BRAIN_PATTERN.search(section) or HEAD_PATTERN.search(section)
            for section in sections
        )
    )

    any_method_section_contains_spine = method_sections_series.apply(
        lambda sections: any(SPINE_PATTERN.search(section) for section in sections)
    )

    has_brain_head_mri_mra_label = dataframe["_report_text"].str.contains(
        BRAIN_HEAD_MRI_MRA_LABEL_PATTERN,
        regex=True,
    )

    no_method_section_contains_brain_or_head = ~any_method_section_contains_brain_or_head

    # Condition 1-1:
    # At least one method section exists,
    # at least one method section mentions spine,
    # and no method section mentions brain/head.
    condition_1_1 = (
        has_method_section
        & any_method_section_contains_spine
        & no_method_section_contains_brain_or_head
    )

    # Condition 1-2:
    # More than 5 distinct vertebral levels,
    # no brain structure term in the full report,
    # and no method section mentions brain/head.
    condition_1_2 = (
        (unique_vertebral_level_counts > 5)
        & (~has_any_brain_structure)
        & no_method_section_contains_brain_or_head
    )

    condition_1_preliminary = condition_1_1 | condition_1_2

    # Hard exclusions:
    # - More than 5 distinct brain structure terms
    # - Explicit brain/head MRI/MRA label pattern
    condition_1_final = (
        condition_1_preliminary
        & (~has_too_many_brain_structures)
        & (~has_brain_head_mri_mra_label)
    )

    spine_filtered_dataframe = dataframe.loc[condition_1_final].copy()
    spine_filtered_dataframe["method_section"] = method_sections_series.loc[condition_1_final].apply(join_method_sections)
    spine_filtered_dataframe["unique_vertebral_level_count"] = unique_vertebral_level_counts.loc[condition_1_final].values
    spine_filtered_dataframe["brain_structure_count"] = brain_structure_counts.loc[condition_1_final].values
    spine_filtered_dataframe["has_method_section"] = has_method_section.loc[condition_1_final].values
    spine_filtered_dataframe["condition_1_1_flag"] = condition_1_1.loc[condition_1_final].values
    spine_filtered_dataframe["condition_1_2_flag"] = condition_1_2.loc[condition_1_final].values
    spine_filtered_dataframe["has_brain_head_mri_mra_label"] = has_brain_head_mri_mra_label.loc[condition_1_final].values

    no_impression_condition = ~dataframe["_report_text"].str.contains(
        IMPRESSION_PATTERN,
        regex=True,
    )

    no_impression_dataframe = dataframe.loc[no_impression_condition].copy()
    no_impression_dataframe["method_section"] = method_sections_series.loc[no_impression_condition].apply(join_method_sections)
    no_impression_dataframe["brain_structure_count"] = brain_structure_counts.loc[no_impression_condition].values
    no_impression_dataframe["has_brain_head_mri_mra_label"] = has_brain_head_mri_mra_label.loc[no_impression_condition].values

    base_output_columns = []
    for column in [id_column, report_column]:
        if column in dataframe.columns and column not in base_output_columns:
            base_output_columns.append(column)

    optional_review_columns = [
        "method_section",
        "unique_vertebral_level_count",
        "brain_structure_count",
        "has_method_section",
        "condition_1_1_flag",
        "condition_1_2_flag",
        "has_brain_head_mri_mra_label",
    ]

    spine_output_columns = build_output_columns(
        spine_filtered_dataframe,
        base_output_columns,
        optional_review_columns,
    )
    no_impression_output_columns = build_output_columns(
        no_impression_dataframe,
        base_output_columns,
        optional_review_columns,
    )

    spine_filtered_dataframe.to_csv(
        OUTPUT_SPINE_FILTERED_PATH,
        index=False,
        columns=spine_output_columns,
    )
    no_impression_dataframe.to_csv(
        OUTPUT_NO_IMPRESSION_PATH,
        index=False,
        columns=no_impression_output_columns,
    )

    print(f"Detected ID column: {id_column}")
    print(f"Detected report text column: {report_column}")
    print(
        "Condition 1 summary:\n"
        "  - Condition 1-1: at least one method section contains 'spine', and all method sections exclude 'brain/head'\n"
        "  - Condition 1-2: more than 5 unique vertebral levels, no brain structure terms, and all method sections exclude 'brain/head'\n"
        "  - Hard exclusions: brain structure count must be <= 5 and no brain/head MRI/MRA label pattern\n"
        f"Total selected: {len(spine_filtered_dataframe)} -> {OUTPUT_SPINE_FILTERED_PATH}"
    )
    print(
        f"Condition 2 summary (reports without 'Impression'): "
        f"{len(no_impression_dataframe)} -> {OUTPUT_NO_IMPRESSION_PATH}"
    )


if __name__ == "__main__":
    main()
