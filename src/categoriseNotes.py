import argparse
import re
import logging
import sys
from typing import List, Tuple
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
    categories = [Category(name=line[len(categoryPrefix):].strip()) for line in category_lines]
    return Categories(categories=categories)

def generate_categories(notes: List[str]) -> Categories:
    prompt = f"""below are notes I have written on a certain topic. provide a list of sub topics which I can use to categorise these notes
- ensure there are sufficient categories to represent depth & breadth of notes
- however also ensure no categories overlap/are redundant
- carefully read the notes to understand the material, and how I personally think about it
- align the categories with how you believe I would conceptually separate the notes in my mind
- the categories should be useful groupings I can use to further develop my notes
- do not try to align the categories with ones from academia, politics and industry
- category names should be:
    - very specific
    - extremely non-generic
    - heavily informed by the contents of the notes
- category names should not contain:
    - a colon or have more than one part/section
    - any fluff/cringe/commentary/hype


Notes:
{' '.join(notes)}"""
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
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
            messages=[{"role": "user", "content": prompt}],
        )

        # logger.debug(f"API Response: {response}")
        
        print("Categories:\n"+"\n".join([cat["name"] for cat in response.content[0].input["categories"]]))

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
    - Do not add or remove any text from the original note.
    - The resulting split pieces must add up exactly to the original note.

    Note to potentially split:
    {note}

    If you decide to split the note, return a list of the split notes. If no split is necessary, return a list containing only the original note."""

    for attempt in range(3):  # Try up to 3 times (original attempt + 2 retries)
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
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
                                    "description": "The list of split notes or the original note if no split is necessary",
                                }
                            },
                            "required": ["split_notes"],
                        },
                    }
                ],
                tool_choice={"type": "tool", "name": "split_note"},
                messages=[{"role": "user", "content": prompt}],
            )

            # logger.debug(f"API Response: {response}")

            if response.content and isinstance(
                response.content[0], anthropic.types.ToolUseBlock
            ):
                split_notes_dict = response.content[0].input
                split_notes = NoteSplit.parse_obj(split_notes_dict).split_notes

                # Verify that the split notes add up to the original note
                splitNoteText = normaliseText('\n'.join(split_notes))
                originalText = normaliseText(note)
                ## remove all non-alphanumeric characters
                if splitNoteText != originalText:
                    print("New:\n"+splitNoteText)
                    print("Old:\n"+originalText)
                    raise ValueError("The split notes do not match the original note exactly.")

                # Verify that all splits occurred on newlines
                original_lines = [normaliseText(line) for line in note.split('\n') if line.strip()]
                original_lines = [line for line in original_lines if line]
                split_lines = [normaliseText(line) for split_note in split_notes for line in split_note.split('\n') if line.strip()]
                split_lines = [line for line in split_lines if line]
                if original_lines != split_lines:
                    print("New:\n"+"\n".join(split_lines))
                    print(split_lines)
                    print("Old:\n"+"\n".join(original_lines))
                    print(original_lines)
                    
                    raise ValueError("Splits did not occur only on newline characters.")
                if len(split_notes) > 1:
                    logger.info(f"Split note:\nINTO {len(split_notes)} notes:\n______________\n"+"\n________________\n".join(split_notes)+"\n______________")
                return split_notes
            else:
                raise ValueError(
                    f"Unexpected response format from Claude API: {response.content}"
                )
        except Exception as e:
            logger.error(f"Error in split_note_if_needed (attempt {attempt + 1}): {e}")
            if attempt < 2:  # If it's not the last attempt
                # Inform Claude about the error
                error_prompt = f"The previous attempt to split the note failed with the following error: {str(e)}. Please try again, paying special attention to the rules and ensuring the output matches the expected format."
                client.messages.create(
                    model="claude-3-5-sonnet-20240620",
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
            model="claude-3-5-sonnet-20240620",
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

        # logger.debug(f"API Response: {response}")

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


def main():
    parser = argparse.ArgumentParser(
        description="Categorize notes from a markdown file."
    )
    parser.add_argument("file_path", help="Path to the markdown file containing notes.")
    parser.add_argument("--split", action="store_true", help="Enable note splitting")
    args = parser.parse_args()

    try:
        front_matter, special_content, notes, category_lines = parse_notes(args.file_path)
        logger.info(f"Parsed {len(notes)} notes from {args.file_path}")
        use_existing = False
        if len(category_lines) > 1:
            categories = extract_existing_categories(category_lines)
            print("Categories:")
            for cat in categories.categories:
                print(f"- {cat.name}")
            use_existing = input("Existing categories found. Use them? (y/n): ").lower() in ["y", "yes"]
            if use_existing:
                logger.info("Using existing categories")

        categoriesRejected = not use_existing
        while categoriesRejected:
            print("Categories:")
            for cat in categories.categories:
                print(f"- {cat.name}")
            categoriesRejected = False if input("Good categories? (y/n): ").lower() in ["y", "yes"] else True
            if categoriesRejected:
                categories = generate_categories(notes)

        categorized_notes = {}
        for i, note in enumerate(notes):
            if args.split and note.count('\n') > 1:
                split_notes = split_note_if_needed(note, categories)
            else:
                split_notes = [note]

            for split_note in split_notes:
                prev_note = notes[i - 1] if i > 0 else ""
                next_note = notes[i + 1] if i < len(notes) - 1 else ""
                category = categorize_note(split_note, prev_note, next_note, categories)
                if category not in categorized_notes:
                    categorized_notes[category] = []
                categorized_notes[category].append(split_note)

        write_categorized_notes(
            args.file_path, front_matter, special_content, categorized_notes
        )

        logger.info("Categorization completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
