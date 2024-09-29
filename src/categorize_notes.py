import argparse
import json
import logging
from typing import List
from pydantic import BaseModel
from openai import OpenAI
from openai.types.chat import ChatCompletion

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

client = OpenAI()


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
        content = "\n".join(lines[1:])

    # Split notes
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
{' '.join(notes)}

Respond with a JSON object containing a 'categories' key with an array of category objects, each containing just a 'name' field."""
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {"role": "user", "content": prompt},
            ],
        )

        response_content = completion.choices[0].message.content
        logger.debug(f"API Response: {response_content}")

        try:
            categories_dict = json.loads(response_content)
            return Categories.parse_obj(categories_dict)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}")
            # Attempt to extract JSON from the response if it's not properly formatted
            try:
                json_start = response_content.index("{")
                json_end = response_content.rindex("}") + 1
                extracted_json = response_content[json_start:json_end]
                categories_dict = json.loads(extracted_json)
                return Categories.parse_obj(categories_dict)
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Failed to extract JSON from response: {e}")
                raise ValueError(
                    f"Failed to parse the API response as JSON. Response: {response_content}"
                )
    except Exception as e:
        logger.error(f"Error in generate_categories: {e}")
        raise


def categorize_note(
    note: str, prev_note: str, next_note: str, categories: Categories
) -> str:
    category_names = [cat.name for cat in categories.categories]
    prompt = f"Categorize the following note into one of these categories: {', '.join(category_names)}. Consider the context provided by the previous and next notes. Respond with a JSON object containing a 'category' field with the chosen category name.\n\nPrevious note:\n{prev_note}\n\nNote to categorize:\n{note}\n\nNext note:\n{next_note}"

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that categorizes notes. Always respond with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        response_content = completion.choices[0].message.content
        logger.debug(f"API Response: {response_content}")

        try:
            category_dict = json.loads(response_content)
            return NoteCategory.parse_obj(category_dict).category
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}")
            # Attempt to extract JSON from the response if it's not properly formatted
            try:
                json_start = response_content.index("{")
                json_end = response_content.rindex("}") + 1
                extracted_json = response_content[json_start:json_end]
                category_dict = json.loads(extracted_json)
                return NoteCategory.parse_obj(category_dict).category
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Failed to extract JSON from response: {e}")
                raise ValueError(
                    f"Failed to parse the API response as JSON. Response: {response_content}"
                )
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
            for note in notes:
                print(f"- {note}")

        logger.info("Categorization completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
