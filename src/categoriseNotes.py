import argparse
import time
import pyperclip
import re
from typing import Callable, Any
from functools import wraps
import logging
import sys
from typing import List, Tuple, Dict
from pydantic import BaseModel
import anthropic
from anthropic import RateLimitError
from helperPrompts import splitPrompt, generateCategoriesPrompt

handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[handler]
)
logger = logging.getLogger(__name__)

client = anthropic.Anthropic()

categoryPrefix = "## -- "

generationModel = "claude-3-opus-20240229"
categorisationModel = "claude-3-5-sonnet-20240620"


class Category(BaseModel):
    name: str


class Categories(BaseModel):
    categories: List[Category]


class NoteCategory(BaseModel):
    category: str


class NoteSplit(BaseModel):
    split_notes: List[str]


def parse_notes(file_path: str) -> Tuple[str, str, List[str], List[str]]:
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
    for line in lines:
        if line.startswith(categoryPrefix):
            category_lines.append(line)
        elif not found_normal_line and (
            line.startswith("$") or (line.startswith("#") and not line.startswith(categoryPrefix))
        ):
            special_lines.append(line)
        else:
            if line.strip():  # If this is a non-empty line
                found_normal_line = True
            normal_lines.append(line)

    special_content = "\n".join(special_lines) + "\n" if special_lines else ""
    content = "\n".join(normal_lines)

    # Filter out lines starting with categoryPrefix
    content_lines = [
        line for line in content.split("\n") if not line.startswith(categoryPrefix)
    ]

    # Rejoin lines and split notes
    content = "\n".join(content_lines)
    notes = [note.strip() for note in content.split("\n\n") if note.strip()]

    return front_matter, special_content, notes, category_lines


def extract_existing_categories(category_lines: List[str]) -> Categories:
    categories = [Category(name=line[len(categoryPrefix):].strip().strip(":")) for line in category_lines]
    return Categories(categories=categories)

def normaliseText(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return text.strip()



def handle_rate_limit(headers: Dict[str, str]) -> None:
    """Log rate limit information and sleep if necessary."""
    logger.warning("Rate limit encountered.")
    
    # Extract and log rate limit information
    for limit_type in ['requests', 'tokens']:
        limit = headers.get(f'anthropic-ratelimit-{limit_type}-limit')
        remaining = headers.get(f'anthropic-ratelimit-{limit_type}-remaining')
        reset = headers.get(f'anthropic-ratelimit-{limit_type}-reset')
        logger.info(f"{limit_type.capitalize()}: {remaining}/{limit} (Reset at: {reset})")

    retry_after = headers.get('retry-after')
    if retry_after:
        sleep_time = int(retry_after) + 1  # Add 1 second as a buffer
        logger.info(f"Sleeping for {sleep_time} seconds before retrying...")
        time.sleep(sleep_time)
    else:
        logger.info("No retry-after header provided. Sleeping for 60 seconds as a precaution.")
        time.sleep(60)
def retry_on_error(max_retries: int = 3) -> Callable:
    """Decorator to retry function on specific errors."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Error in {func.__name__} (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)  # Wait a second before retrying
            raise RuntimeError(f"Max retries ({max_retries}) reached in {func.__name__}")
        return wrapper
    return decorator


def retry_on_rate_limit(max_retries: int = None) -> Callable:
    """Decorator to retry function on rate limit error."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            while max_retries is None or retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    retries += 1
                    logger.warning(f"Rate limit hit. Retry attempt {retries}")
                    handle_rate_limit(e.response.headers)
            raise RuntimeError(f"Max retries ({max_retries}) reached due to rate limiting.")
        return wrapper
    return decorator

@retry_on_rate_limit()
@retry_on_error()
def split_note_if_needed(note: str, categories: Categories) -> List[str]:
    category_names = [cat.name for cat in categories.categories]
    prompt = f"""Split the following note into multiple notes if necessary to properly categorize them into the available categories: {', '.join(category_names)}. 
    
    {splitPrompt}

    Note to potentially split:
    {note}

    If you decide to split the note, return a list of the split notes. If no split is necessary, return a list containing only the original note."""

    response = client.messages.create(
        model=categorisationModel,
        max_tokens=1024,
        tools=[{
            "name": "split_note",
            "description": "Split a note into multiple notes if necessary",
            "input_schema": {
                "type": "object",
                "properties": {
                    "split_notes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The list of split notes or a list containing just the original note if no split is necessary",
                    }
                },
                "required": ["split_notes"],
            },
        }],
        tool_choice={"type": "tool", "name": "split_note"},
        messages=[{"role": "user", "content": prompt}],
    )

    if not response.content or not isinstance(response.content[0], anthropic.types.ToolUseBlock):
        raise ValueError(f"Unexpected response format from Claude API: {response.content}")

    split_notes_dict = response.content[0].input
    split_notes = NoteSplit.model_validate(split_notes_dict).split_notes

    # Validate the split
    validate_split_notes(note, split_notes)

    if len(split_notes) > 1:
        logger.info(f"Split one note into {len(split_notes)} notes:\n" + "\n".join(split_notes))

    return split_notes

def validate_split_notes(original_note: str, split_notes: List[str]) -> None:
    """Validate that the split notes match the original note."""
    split_text = normaliseText('\n'.join(split_notes))
    original_text = normaliseText(original_note)
    
    if split_text != original_text:
        logger.error(f"Split text:\n{split_text}")
        logger.error(f"Original text:\n{original_text}")
        raise ValueError("The split notes do not match the original note exactly.")

    original_lines = [line for line in original_note.split('\n') if line.strip()]
    split_lines = [line for note in split_notes for line in note.split('\n') if line.strip()]
    
    if original_lines != split_lines:
        logger.error(f"Split lines:\n{split_lines}")
        logger.error(f"Original lines:\n{original_lines}")
        raise ValueError("Splits did not occur only on newline characters.")

@retry_on_rate_limit()
@retry_on_error()
def categorize_note(note: str, prev_note: str, next_note: str, categories: Categories) -> str:
    category_names = [cat.name for cat in categories.categories]
    prompt = f"""Categorize the following note into one of these categories: {', '.join(category_names)}. 
    Consider the context provided by the previous and next notes.

    Previous note:
    {prev_note}

    Note to categorize:
    {note}

    Next note:
    {next_note}"""

    response = client.messages.create(
        model=categorisationModel,
        max_tokens=1024,
        tools=[{
            "name": "categorize_note",
            "description": "Categorize a note into one of the given categories",
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The chosen category name",
                    }
                },
                "required": ["category"],
            },
        }],
        tool_choice={"type": "tool", "name": "categorize_note"},
        messages=[{"role": "user", "content": prompt}],
    )

    if not response.content or not isinstance(response.content[0], anthropic.types.ToolUseBlock):
        raise ValueError(f"Unexpected response format from Claude API: {response.content}")

    category_dict = response.content[0].input
    category = NoteCategory.parse_obj(category_dict).category

    if category not in category_names:
        raise ValueError(f"Invalid category: {category}. Must be one of: {', '.join(category_names)}")

    return category


def write_categorized_notes(
    file_path: str, front_matter: str, special_content: str, categorized_notes: dict
):
    with open(file_path, "w") as file:
        file.write(front_matter)
        file.write(special_content)
        for category, notes in categorized_notes.items():
            file.write(f"\n\n\n\n\n\n\n\n\n{categoryPrefix}{category}:\n\n")
            file.write("\n\n".join(notes))
    logger.info(f"Categorized notes written back to {file_path}")


def get_user_choice(prompt: str, options: List[str]) -> str:
    while True:
        choice = input(f"{prompt} ({'/'.join(options)}): ").lower()
        if choice in options:
            return choice
        print(f"Invalid choice. Please choose from {', '.join(options)}.")

def edit_categories(categories: Categories) -> Categories:
    categories_str = "\n".join([f"- {cat.name}" for cat in categories.categories])
    copyList = input("Copy current list to your clipboard, to edit? (y/n): ").lower() in ["y", "yes"]
    if copyList:
        pyperclip.copy(categories_str)
    print("Edit above listed categories, then copy the updated list to your clipboard, in the same format as they are listed above. Press Enter when done.")
    input()
    edited_categories_str = pyperclip.paste()
    
    # Parse the edited categories
    new_categories = [
        Category(name=line.strip()[2:])  # Remove the "- " prefix
        for line in edited_categories_str.split('\n')
        if line.strip().startswith('- ')
    ]
    
    return Categories(categories=new_categories)

def display_categories(categories, source):
    print(f"\n{source} categories:")
    for cat in categories.categories:
        print(f"- {cat.name}")

def generate_categories(notes, existing_categories=None, change_description=None):
    prompt = f"""{generateCategoriesPrompt}
    
Notes:
{' '.join(notes)}"""

    messages = [{"role": "user", "content": prompt}]

    if existing_categories and change_description:
        existing_categories_str = "\n".join([f"- {cat.name}" for cat in existing_categories.categories])
        messages.append({
            "role": "assistant",
            "content": f"Here are the categories I've generated based on your notes:\n{existing_categories_str}"
        })
        
        messages.append({
            "role": "user",
            "content": f"Please modify the categories based on this description: {change_description}"
        })

    try:
        response = client.messages.create(
            model=generationModel,
            max_tokens=1024,
            tools=[
                {
                    "name": "generate_categories",
                    "description": "Generate categories for a set of notes",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "categories": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Name of the category",
                                        }
                                    },
                                    "required": ["name"],
                                },
                            }
                        },
                        "required": ["categories"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "generate_categories"},
            messages=messages,
        )
        if response.content and isinstance(
            response.content[0], anthropic.types.ToolUseBlock
        ):
            categories_dict = response.content[0].input
            return Categories.parse_obj(categories_dict)
        else:
            raise ValueError(
                f"Unexpected response format from Claude API: {response.content}"
            )
    except Exception as e:
        logger.error(f"Error in generate_categories: {e}")
        raise

def process_categories(notes, existing_categories=None):
    categories = Categories(categories=[cat for cat in existing_categories.categories]) if existing_categories else None
    source = "Existing" if existing_categories else "Generated"
    
    while True:
        if categories is None:
            categories = generate_categories(notes)
            source = "Generated"
        
        display_categories(categories, source)
        
        choice = get_user_choice("What would you like to do with these categories?", ["keep", "edit", "revise", "new"])
        if choice == "keep":
            return categories
        elif choice == "edit":
            categories = edit_categories(categories)
            source = "Edited"
        elif choice == "revise":
            modification_description = input("Please describe how you'd like to modify the categories: ")
            categories = generate_categories(notes, existing_categories=categories, change_description=modification_description)
            source = "Modified"
        else:  # "new"
            categories = None  # This will trigger generation of new categories in the next iteration

    return categories

def categorize_notes(notes, categories, split):
    categorized_notes = {}
    for i, note in enumerate(notes):
        split_notes = split_note_if_needed(note, categories) if split and note.count('\n') > 1 else [note]
        for split_note in split_notes:
            prev_note = notes[i - 1] if i > 0 else ""
            next_note = notes[i + 1] if i < len(notes) - 1 else ""
            category = categorize_note(split_note, prev_note, next_note, categories)
            categorized_notes.setdefault(category, []).append(split_note)
    return categorized_notes

def main():
    parser = argparse.ArgumentParser(description="Categorize notes from a markdown file.")
    parser.add_argument("file_path", help="Path to the markdown file containing notes.")
    parser.add_argument("--split", action="store_true", help="Enable note splitting")
    args = parser.parse_args()

    try:
        front_matter, special_content, notes, category_lines = parse_notes(args.file_path)
        logger.info(f"Parsed {len(notes)} notes from {args.file_path}")

        existing_categories = extract_existing_categories(category_lines) if len(category_lines) > 1 else None
        categories = process_categories(notes, existing_categories)
        
        categorized_notes = categorize_notes(notes, categories, args.split)
        
        write_categorized_notes(args.file_path, front_matter, special_content, categorized_notes)
        logger.info("Categorization completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()