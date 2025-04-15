from dataclasses import dataclass, field
from datetime import datetime, timezone
from dotenv import load_dotenv
from functools import wraps
from helperPrompts import splitPrompt, generateCategoriesPrompt
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
from typing import List, Tuple, Callable, Any, Optional
import argparse
import http.client
import json
import logging
import os
import pyperclip
import re
import sys
import time


class HTTPFilter(logging.Filter):
    def filter(self, record):
        return not (
            "HTTP Request" in record.getMessage() or "200 OK" in record.getMessage()
        )


handler = logging.StreamHandler(sys.stdout)
handler.addFilter(HTTPFilter())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[handler],
)
logger = logging.getLogger(__name__)

# Disable HTTP connection debugging
http.client.HTTPConnection.debuglevel = 0

# Set logging level for potentially noisy libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

categoryHeadingPrefix = "# -- "
NOTE_DIVIDER = "# -- SCRATCHPAD"


def is_note_divider(line: str) -> bool:
    """Check if a line is a note divider (matches the note divider format or legacy format)"""
    line = line.strip()
    if line == NOTE_DIVIDER:
        return True
    # Legacy divider check (for backward compatibility)
    return line.count("_") >= 6


generationModel = "openai/o3-mini-high"
categorisationModel = "openai/gpt-4o-2024-11-20"


@dataclass
class RetryContext:
    attempts: int = 0
    errors: List[Exception] = field(default_factory=list)


class NoteCategory(BaseModel):
    category_number: int


class NoteSplit(BaseModel):
    split_notes: List[str]


class Category(BaseModel):
    name: str


class Categories(BaseModel):
    categories: List[Category]


@dataclass
class Note:
    content: str
    category: Optional[str] = None


def parse_notes(file_path: str) -> Tuple[str, str, List[Note], List[str]]:
    with open(file_path, "r") as file:
        content = file.read()

    # Extract front matter
    parts = content.split("---", 2)
    front_matter = "---" + parts[1] + "---\n" if len(parts) >= 3 else ""
    content = parts[2].strip() if len(parts) >= 3 else content

    # Extract lines starting with $ or single #
    lines = content.split("\n")
    special_lines = []
    normal_lines = []
    category_lines = []
    found_normal_line = False

    # Track current category and whether we're below divider
    below_divider = False

    for line in lines:
        if is_note_divider(line):
            below_divider = True
            normal_lines.append(line)
            continue

        if line.startswith(categoryHeadingPrefix):
            category_lines.append(line)
            found_normal_line = True
            normal_lines.append(line)
        elif not found_normal_line and (
            line.startswith("$")
            or (line.startswith("#") and not line.startswith(categoryHeadingPrefix))
        ):
            special_lines.append(line)
        else:
            if line.strip():  # If this is a non-empty line
                found_normal_line = True
            normal_lines.append(line)

    special_content = "\n".join(special_lines) + "\n" if special_lines else ""
    content = "\n".join(normal_lines)

    # Split into notes and track categories
    raw_notes = []
    current_note_lines = []
    current_note_category = None
    below_divider = False

    # purpose of the below is to split text on empty lines, category headers and dividers
    for line in content.split("\n"):
        if is_note_divider(line):
            if current_note_lines:
                note_content = "\n".join(current_note_lines).strip()
                if note_content:
                    raw_notes.append(
                        Note(
                            content=note_content,
                            category=None if below_divider else current_note_category,
                        )
                    )
            current_note_lines = []
            below_divider = True
        elif line.startswith(categoryHeadingPrefix):
            if current_note_lines:
                note_content = "\n".join(current_note_lines).strip()
                if note_content:
                    raw_notes.append(
                        Note(
                            content=note_content,
                            category=None if below_divider else current_note_category,
                        )
                    )
            current_note_lines = []
            current_note_category = (
                line[len(categoryHeadingPrefix) :].strip().strip(":")
            )
        elif line.strip():
            current_note_lines.append(line)
        elif current_note_lines:  # Empty line after note content
            note_content = "\n".join(current_note_lines).strip()
            if note_content:
                raw_notes.append(
                    Note(
                        content=note_content,
                        category=None if below_divider else current_note_category,
                    )
                )
            current_note_lines = []

    # Add final note if exists
    if current_note_lines:
        note_content = "\n".join(current_note_lines).strip()
        if note_content:
            raw_notes.append(
                Note(
                    content=note_content,
                    category=None if below_divider else current_note_category,
                )
            )
    notes = [note for note in raw_notes if note.content.strip()]
    return front_matter, special_content, notes, category_lines


def extract_existing_categories(category_lines: List[str]) -> Categories:
    categories = [
        Category(name=line[len(categoryHeadingPrefix) :].strip().strip(":"))
        for line in category_lines
    ]
    return Categories(categories=categories)


def normaliseText(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-zA-Z0-9]", "", text)
    return text.strip()


def validate_split_notes(original_note: str, split_notes: List[str]) -> None:
    """Validate that the split notes match the original note."""
    split_text = normaliseText("\n".join(split_notes))
    original_text = normaliseText(original_note)

    if split_text != original_text:
        logger.error(f"Split text:\n{split_text}")
        logger.error(f"Original text:\n{original_text}")
        raise ValueError(
            "The split notes do not match the original note exactly. The resulting split pieces must add up exactly to the original note. i.e. do not add or remove ANY text from the original note, or re-order the text!"
        )

    original_lines = [line for line in original_note.split("\n") if line.strip()]
    split_lines = [
        line for note in split_notes for line in note.split("\n") if line.strip()
    ]

    if original_lines != split_lines:
        logger.error(f"Split lines:\n{split_lines}")
        logger.error(f"Original lines:\n{original_lines}")
        raise ValueError(
            "Splits did not occur only on newline characters. Splits must only be made on newline characters."
        )


def retry_on_error(max_retries: int = 3) -> Callable:
    """Decorator to retry function on specific errors, passing error context."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retry_context = RetryContext()

            for attempt in range(max_retries):
                retry_context.attempts = attempt + 1
                try:
                    return func(*args, **kwargs, retry_context=retry_context)
                except (ValueError, RuntimeError) as e:
                    logger.error(
                        f"Error in {func.__name__} (attempt {attempt + 1}): {e}"
                    )
                    retry_context.errors.append(e)
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)  # Wait a second before retrying
            raise RuntimeError(
                f"Max retries ({max_retries}) reached in {func.__name__}"
            )

        return wrapper

    return decorator


@retry_on_error()
def split_note_if_needed(
    note: str, categories: Categories, retry_context: Optional[RetryContext] = None
) -> List[str]:
    category_names = [cat.name for cat in categories.categories]

    error_context = ""
    if retry_context and retry_context.errors:
        error_context = (
            f"\n\nPrevious attempts failed with the following errors:\n"
            + "\n".join(
                f"Attempt {i+1}: {str(e)}" for i, e in enumerate(retry_context.errors)
            )
        )

    prompt = f"""Split the following note into multiple notes if necessary to properly categorize them into the available categories: {', '.join(category_names)}. 
    
    {splitPrompt}

    Note to potentially split:
    {note}
    
    If you decide to split the note, return a list of the split notes. If no split is necessary, return a list containing only the original note.
    
    {error_context}
    """

    response = client.beta.chat.completions.parse(
        model=categorisationModel,
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at splitting notes into appropriate categories.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=NoteSplit,
    )

    split_notes = response.choices[0].message.parsed.split_notes

    # Validate the split
    validate_split_notes(note, split_notes)

    if len(split_notes) > 1:
        logger.info(
            f"Split one note into {len(split_notes)} notes:\n" + "\n".join(split_notes)
        )

    return split_notes


@retry_on_error()
def categorize_note(
    note: str,
    prev_note: str,
    next_note: str,
    categories: Categories,
    retry_context: Optional[RetryContext] = None,
) -> str:
    category_list = [cat.name for cat in categories.categories]

    error_context = ""
    if retry_context and retry_context.errors:
        error_context = (
            f"\n\nPrevious attempts failed with the following errors:\n"
            + "\n".join(
                f"Attempt {i+1}: {str(e)}" for i, e in enumerate(retry_context.errors)
            )
        )

    prompt = f"""Categorize the following note into one of these numbered categories:
{'\n'.join([f"{i+1}. {name}" for i, name in enumerate(category_list)])}

Note to categorize:
{note}

{error_context}

Respond with ONLY the number of the chosen category.
"""

    response = client.beta.chat.completions.parse(
        model=categorisationModel,
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at categorizing notes into predefined categories.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=NoteCategory,
    )

    category_number = response.choices[0].message.parsed.category_number

    if category_number < 1 or category_number > len(category_list):
        raise ValueError(
            f"Invalid category number: {category_number}. Must be between 1 and {len(category_list)}"
        )

    return category_list[category_number - 1]


def write_categorized_notes(
    file_path: str,
    front_matter: str,
    special_content: str,
    categorized_notes: dict,
    categories: Categories,
):
    with open(file_path, "w") as file:
        file.write(front_matter)
        file.write(special_content)
        file.write("\n\n")
        # Iterate through categories in their original order
        for category_obj in categories.categories:
            category_name = category_obj.name
            notes = categorized_notes.get(category_name, [])
            file.write(f"{categoryHeadingPrefix}{category_name}:\n\n")
            file.write("\n\n".join(notes) + "\n\n\n\n\n\n\n\n\n")
        file.write(f"\n{NOTE_DIVIDER}\n")


def get_user_choice(prompt: str, options: List[str]) -> str:
    while True:
        choice = input(f"{prompt} ({'/'.join(options)}): ").lower()
        if choice in options:
            return choice
        print(f"Invalid choice. Please choose from {', '.join(options)}.")


def edit_categories(categories: Categories) -> Categories:
    categories_str = "\n".join([f"- {cat.name}" for cat in categories.categories])
    copyList = input(
        "Copy current list to your clipboard, to edit? (y/n): "
    ).lower() in ["y", "yes"]
    if copyList:
        pyperclip.copy(categories_str)
    print(
        "Edit above listed categories, then copy the updated list to your clipboard, in the same format as they are listed above. Press Enter when done."
    )
    input()
    edited_categories_str = pyperclip.paste()

    # Parse the edited categories
    new_categories = [
        Category(name=line.strip()[2:])  # Remove the "- " prefix
        for line in edited_categories_str.split("\n")
        if line.strip().startswith("- ")
    ]

    return Categories(categories=new_categories)


def display_categories(categories, source):
    print(f"\n{source} categories:")
    for cat in categories.categories:
        print(f"- {cat.name}")


def generate_categories(notes, existing_categories=None, change_description=None):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_categories",
                "description": "Create a list of categories based on the given notes",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                },
                                "required": ["name"],
                            },
                        }
                    },
                    "required": ["categories"],
                },
            },
        }
    ]

    prompt = f"""{generateCategoriesPrompt}\n\n Notes:\n{'\n\n'.join([note.content for note in notes])}"""

    messages = [
        {
            "role": "system",
            "content": "You are an expert at generating categories for a set of notes.",
        },
        {"role": "user", "content": prompt},
    ]

    if existing_categories and change_description:
        existing_categories_str = "\n".join(
            [f"- {cat.name}" for cat in existing_categories.categories]
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"Here are the categories I've generated based on your notes:\n{existing_categories_str}",
            }
        )

        messages.append(
            {
                "role": "user",
                "content": f"Please modify the categories based on this description: {change_description}",
            }
        )

    try:
        response = client.chat.completions.create(
            model=generationModel,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_categories"}},
        )

        # Extract the function call from the response
        function_call = response.choices[0].message.tool_calls[0].function
        categories_data = json.loads(function_call.arguments)

        # Convert the categories data to your Categories object
        return Categories(
            categories=[Category(**cat) for cat in categories_data["categories"]]
        )
    except Exception as e:
        logger.error(
            f"Error in generate_categories: {e}, messages: {messages}, response: {response}"
        )
        raise


def process_categories(notes, existing_categories=None):
    categories = (
        Categories(categories=[cat for cat in existing_categories.categories])
        if existing_categories
        else None
    )
    source = "Existing" if existing_categories else "Generated"

    while True:
        if categories is None:
            categories = generate_categories(notes)
            source = "Generated"

        display_categories(categories, source)

        choice = get_user_choice(
            "What would you like to do with these categories?",
            ["done", "edit", "revise", "new"],
        )
        if choice == "done":
            return categories
        elif choice == "edit":
            categories = edit_categories(categories)
            source = "Edited"
        elif choice == "revise":
            modification_description = input(
                "Please describe how you'd like to modify the categories: "
            )
            categories = generate_categories(
                notes,
                existing_categories=categories,
                change_description=modification_description,
            )
            source = "Modified"
        elif choice == "new":
            categories = None

    return categories


def categorize_notes(notes: List[Note], categories: Categories, split: bool) -> dict:
    categorized_notes = {}
    category_names = [cat.name for cat in categories.categories]

    # Wrap the notes list with tqdm for progress tracking
    for i, note in enumerate(tqdm(notes, desc="Categorizing notes", unit="note")):
        # If note has an existing category that's still valid, use it
        if note.category and note.category in category_names:
            categorized_notes.setdefault(note.category, []).append(note.content)
            continue

        # Otherwise, process the note
        split_notes = (
            split_note_if_needed(note.content, categories)
            if split and note.content.count("\n") >= 1
            else [note.content]
        )
        for split_note in split_notes:
            prev_note = notes[i - 1].content if i > 0 else ""
            next_note = notes[i + 1].content if i < len(notes) - 1 else ""
            category = categorize_note(split_note, prev_note, next_note, categories)
            categorized_notes.setdefault(category, []).append(split_note)

    return categorized_notes


def main():
    parser = argparse.ArgumentParser(
        description="Categorize notes from a markdown file."
    )
    parser.add_argument("file_path", help="Path to the markdown file containing notes.")
    parser.add_argument("--split", action="store_true", help="Enable note splitting")
    args = parser.parse_args()

    try:
        front_matter, special_content, notes, category_lines = parse_notes(
            args.file_path
        )
        logger.info(f"Parsed {len(notes)} notes from {args.file_path}")

        existing_categories = (
            extract_existing_categories(category_lines)
            if len(category_lines) > 1
            else None
        )
        categories = process_categories(notes, existing_categories)

        categorized_notes = categorize_notes(notes, categories, args.split)

        write_categorized_notes(
            args.file_path, front_matter, special_content, categorized_notes, categories
        )
        logger.info("Categorization completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
