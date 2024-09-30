import argparse
import pyperclip
import re
import logging
import sys
from typing import List, Tuple, Dict, Optional
from pydantic import BaseModel
import anthropic

handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[handler]
)
logger = logging.getLogger(__name__)

client = anthropic.Anthropic()

categoryPrefix = "## -- "

model = "claude-3-opus-20240229"
# model = "claude-3-5-sonnet-20240620"


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
    front_matter = "---\n" + parts[1] + "---\n" if len(parts) >= 3 else ""
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

def split_note_if_needed(note: str, categories: Categories) -> List[str]:
    category_names = [cat.name for cat in categories.categories]
    prompt = f"""Split the following note into multiple notes if necessary to properly categorize them into the available categories: {', '.join(category_names)}. 
    
    Rules:
    - Only split a note if BOTH of the below conditions are met:
        - the note contains meaningfully distinct sub-parts which very clearly do not all belong under a single one of the above categories
        - the resulting split notes make sense in isolation, can be understood independently and do not depend on each other for context
    - Splits must only occur on newline characters.
    - Do not just split a note just because it has some kind of dividers/sub-sections in it. Only split it if it has sub sections which actually belong in seperate categories!
    - Do not split in the middle of a line of text.
    - DO NOT add or remove ANY text from the original note!!
    - The resulting split pieces must add up exactly to the original note.

    Note to potentially split:
    {note}

    If you decide to split the note, return a list of the split notes. If no split is necessary, return a LIST (not merely a string) containing only the original note."""

    for attempt in range(3):  # Try up to 3 times (original attempt + 2 retries)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                tools=[
                    {
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
                    }
                ],
                tool_choice={"type": "tool", "name": "split_note"},
                messages=[{"role": "user", "content": prompt}],
            )

            if response.content and isinstance(
                response.content[0], anthropic.types.ToolUseBlock
            ):
                split_notes_dict = response.content[0].input
                if type(split_notes_dict["split_notes"]) == str:
                    split_notes_dict["split_notes"] = split_notes_dict["split_notes"].split("\n\n")
                split_notes = NoteSplit.model_validate(split_notes_dict).split_notes

                # Verify that the split notes add up to the original note
                splitNoteText = normaliseText('\n'.join(split_notes))
                originalText = normaliseText(note)
                ## remove all non-alphanumeric characters
                if splitNoteText != originalText:
                    logger.error("New:\n"+splitNoteText)
                    logger.error("Old:\n"+originalText)
                    raise ValueError("The split notes do not match the original note exactly.")

                # Verify that all splits occurred on newlines
                original_lines = [normaliseText(line) for line in note.split('\n') if line.strip()]
                original_lines = [line for line in original_lines if line]
                split_lines = [normaliseText(line) for split_note in split_notes for line in split_note.split('\n') if line.strip()]
                split_lines = [line for line in split_lines if line]
                if original_lines != split_lines:
                    logger.error("New:\n"+"\n".join(split_lines))
                    logger.error(split_lines)
                    logger.error("Old:\n"+"\n".join(original_lines))
                    logger.error(original_lines)
                    raise ValueError("Splits did not occur only on newline characters.")
                if len(split_notes) > 1:
                    logger.info(f"Split one note into {len(split_notes)} notes:\n______________\n"+"\n________________\n".join(split_notes)+"\n______________")
                return split_notes
            else:
                raise ValueError(
                    f"Unexpected response format from Claude API: {response.content}"
                )
        except Exception as e:
            logger.error(f"Error in split_note_if_needed (attempt {attempt + 1}): {e}\nSplit notes dict: {split_notes_dict}")
            if attempt < 2:  # If it's not the last attempt
                # Inform Claude about the error
                error_prompt = f"The previous attempt to split the note failed with the following error: {str(e)}. Please try again, paying special attention to the rules and ensuring the output matches the expected format."
                logger.error("Error prompt: "+error_prompt)
                client.messages.create(
                    model=model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": error_prompt}],
                )
            else:
                logger.warning("Max retries reached. Returning original note.")
                return [note]  # Return the original note if all attempts fail

    # This line should never be reached due to the return in the else block above,
    # but we include it for completeness
    return [note]


def categorize_note(
    note: str, prev_note: str, next_note: str, categories: Categories
) -> str:
    category_names = [cat.name for cat in categories.categories]
    prompt = f"Categorize the following note into one of these categories: {', '.join(category_names)}. Consider the context provided by the previous and next notes.\n\nPrevious note:\n{prev_note}\n\nNote to categorize:\n{note}\n\nNext note:\n{next_note}"

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=[
                {
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
                }
            ],
            tool_choice={"type": "tool", "name": "categorize_note"},
            messages=[{"role": "user", "content": prompt}],
        )

        if response.content and isinstance(
            response.content[0], anthropic.types.ToolUseBlock
        ):
            category_dict = response.content[0].input
            return NoteCategory.parse_obj(category_dict).category
        else:
            raise ValueError(
                f"Unexpected response format from Claude API: {response.content}"
            )
    except Exception as e:
        logger.error(f"Error in categorize_note: {e}")
        raise


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
    pyperclip.copy(categories_str)
    print("Categories copied to clipboard. Edit then copy them to your clipboard. Press Enter when done.")
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
    prompt = f"""Below are notes I have written on a certain topic. Provide a list of sub topics which I can use to categorise these notes
- Ensure there are sufficient categories to represent depth & breadth of notes
- However also ensure no categories overlap/are redundant
- Carefully read the notes to understand the material, and how I personally think about it
- Align the categories with how you believe I would conceptually separate the notes in my mind
- The categories should be useful groupings I can use to further develop my notes
- Do not try to align the categories with ones from academia, politics and industry
- Category names should be:
    - very specific
    - extremely non-generic
    - heavily informed by the contents of the notes
- Category names should not contain:
    - a colon or have more than one part/section
    - any fluff/cringe/commentary/hype
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
            model=model,
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