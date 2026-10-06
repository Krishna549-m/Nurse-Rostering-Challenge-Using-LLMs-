from pathlib import Path
from openai import OpenAI
from config.llm_config import call_llm


# Files
PROJECT_DIR = Path(__file__).resolve().parents[1]

PROMPT1_FILE = PROJECT_DIR/ "final_prompts/initial_prompt_5_1.txt"
PROMPT2_FILE = PROJECT_DIR/ "final_prompts/initial_prompt_5_2.txt"
RAW_INPUT_FILE = PROJECT_DIR/ "final_prompts/raw_input.txt"

PROMPT1_BUILT = PROJECT_DIR/ "built_prompts/prompt1_built.txt"
PROMPT2_BUILT = PROJECT_DIR/ "built_prompts/prompt2_built.txt"

RESPONSE1_FILE = PROJECT_DIR/ "json_output/response1.json"
RESPONSE2_FILE = PROJECT_DIR/ "source/solver.py"


def clean_json(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines)

    return text.strip()


def clean_python(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines)

    return text.strip()




def main():

    # -----------------------------
    # PROMPT 1
    # -----------------------------
    prompt1_template = Path(PROMPT1_FILE).read_text(encoding="utf-8")
    raw_text = Path(RAW_INPUT_FILE).read_text(encoding="utf-8")

    prompt1 = prompt1_template.replace("{data}", raw_text)

    Path(PROMPT1_BUILT).write_text(prompt1, encoding="utf-8")

    print("Running Prompt 1...")
    response1 = call_llm(prompt1)

    response1 = clean_json(response1)
    Path(RESPONSE1_FILE).write_text(response1, encoding="utf-8")

    print("Prompt 1 done.")

    # -----------------------------
    # PROMPT 2
    # -----------------------------
    prompt2_template = Path(PROMPT2_FILE).read_text(encoding="utf-8")
    prompt2 = prompt2_template.replace("{data}", response1)

    Path(PROMPT2_BUILT).write_text(prompt2, encoding="utf-8")

    print("Running Prompt 2...")
    response2 = call_llm(prompt2)

    response2 = clean_python(response2)
    Path(RESPONSE2_FILE).write_text(response2, encoding="utf-8")

    print("Prompt 2 done.")


if __name__ == "__main__":
    main()