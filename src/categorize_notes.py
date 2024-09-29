import argparse
import json
import logging
from typing import List
from pydantic import BaseModel
import anthropic

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

client = anthropic.Anthropic()


class Category(BaseModel):
    name: str


class Categories(BaseModel):
    categories: List[Category]


class NoteCategory(BaseModel):
    category: str


def parse_notes(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        content = file.read()

    # Skip front matter
    parts = content.split("---", 2)
    if len(parts) >= 3:
        content = parts[2].strip()

    # Skip the first line if it starts with $
    lines = content.split("\n")
    if lines and lines[0].startswith("$"):
        lines = lines[1:]

    # Filter out lines starting with ########
    lines = [line for line in lines if not line.startswith("########")]

    # Rejoin lines and split notes
    content = "\n".join(lines)
    return [note.strip() for note in content.split("\n\n") if note.strip()]


def generate_categories(notes: List[str]) -> Categories:
    prompt = f"""below are notes I have written on a certain topic
- provide a list of sub topics which I can use to categorise these notes
- no sub dot points or descriptions
- ensure no categories overlap
- carefully read the notes to understand the material, and how I personally think about it
- align the categories with how you believe I would conceptually separate the notes, not how such topics are normally categorised in e.g. academia and industry
- make the categories as specific as possible without increasing their quantity
- the name of each category should be extremely non-generic & heavily informed by the contents of the notes

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
                                            "description": "Name of the category"
                                        }
                                    },
                                    "required": ["name"]
                                }
                            }
                        },
                        "required": ["categories"]
                    }
                }
            ],
            tool_choice={"type": "tool", "name": "generate_categories"},
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        logger.debug(f"API Response: {response}")

        if response.content and isinstance(response.content[0], anthropic.types.ToolUseBlock):
            categories_dict = json.loads(response.content[0].input)
            return Categories.parse_obj(categories_dict)
        else:
            raise ValueError(f"Unexpected response format from Claude API: {response.content}")
    except Exception as e:
        logger.error(f"Error in generate_categories: {e}")
        raise


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
                                "description": "The chosen category name"
                            }
                        },
                        "required": ["category"]
                    }
                }
            ],
            tool_choice={"type": "tool", "name": "categorize_note"},
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        logger.debug(f"API Response: {response}")

        if response.content and isinstance(response.content[0], anthropic.types.ToolUseBlock):
            category_dict = json.loads(response.content[0].input)
            return NoteCategory.parse_obj(category_dict).category
        else:
            raise ValueError(f"Unexpected response format from Claude API: {response.content}")
    except Exception as e:
        logger.error(f"Error in categorize_note: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Categorize notes from a markdown file."
    )
    parser.add_argument("file_path", help="Path to the markdown file containing notes.")
    args = parser.parse_args()

    try:
        notes = parse_notes(args.file_path)
        logger.info(f"Parsed {len(notes)} notes from {args.file_path}")

        categories = generate_categories(notes)
        logger.info(f"Generated {len(categories.categories)} categories")

        categorized_notes = {}
        for i, note in enumerate(notes):
            prev_note = notes[i - 1] if i > 0 else ""
            next_note = notes[i + 1] if i < len(notes) - 1 else ""
            category = categorize_note(note, prev_note, next_note, categories)
            if category not in categorized_notes:
                categorized_notes[category] = []
            categorized_notes[category].append(note)

        for category, notes in categorized_notes.items():
            print(f"\n\n\n########{category}:")
            print("\n\n".join(notes))

        logger.info("Categorization completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
