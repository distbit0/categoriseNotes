import argparse
import json
from typing import List
from pydantic import BaseModel
from openai import OpenAI
from openai.types.chat import ChatCompletion

client = OpenAI()


class Category(BaseModel):
    name: str
    description: str


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
    prompt = f"Based on the following notes, generate a list of categories that best represent the content. Each category should have a name and a brief description. Respond with a JSON object containing a 'categories' key with an array of category objects, each having 'name' and 'description' fields:\n\n{' '.join(notes)}"

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that categorizes notes. Always respond with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    response_content = completion.choices[0].message.content
    print(f"API Response: {response_content}")  # Debug print
    try:
        categories_dict = json.loads(response_content)
        return Categories.parse_obj(categories_dict)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        # Attempt to extract JSON from the response if it's not properly formatted
        try:
            json_start = response_content.index('{')
            json_end = response_content.rindex('}') + 1
            extracted_json = response_content[json_start:json_end]
            categories_dict = json.loads(extracted_json)
            return Categories.parse_obj(categories_dict)
        except (ValueError, json.JSONDecodeError):
            raise ValueError(f"Failed to parse the API response as JSON. Response: {response_content}")


def categorize_note(
    note: str, prev_note: str, next_note: str, categories: Categories
) -> str:
    category_names = [cat.name for cat in categories.categories]
    prompt = f"Categorize the following note into one of these categories: {', '.join(category_names)}. Consider the context provided by the previous and next notes. Respond with a JSON object containing a 'category' field with the chosen category name.\n\nPrevious note:\n{prev_note}\n\nNote to categorize:\n{note}\n\nNext note:\n{next_note}"

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that categorizes notes. Always respond with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    response_content = completion.choices[0].message.content
    print(f"API Response: {response_content}")  # Debug print
    try:
        category_dict = json.loads(response_content)
        return NoteCategory.parse_obj(category_dict).category
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        # Attempt to extract JSON from the response if it's not properly formatted
        try:
            json_start = response_content.index('{')
            json_end = response_content.rindex('}') + 1
            extracted_json = response_content[json_start:json_end]
            category_dict = json.loads(extracted_json)
            return NoteCategory.parse_obj(category_dict).category
        except (ValueError, json.JSONDecodeError):
            raise ValueError(f"Failed to parse the API response as JSON. Response: {response_content}")


def main():
    parser = argparse.ArgumentParser(
        description="Categorize notes from a markdown file."
    )
    parser.add_argument("file_path", help="Path to the markdown file containing notes.")
    args = parser.parse_args()

    notes = parse_notes(args.file_path)
    categories = generate_categories(notes)

    categorized_notes = {}
    for i, note in enumerate(notes):
        prev_note = notes[i - 1] if i > 0 else ""
        next_note = notes[i + 1] if i < len(notes) - 1 else ""
        category = categorize_note(note, prev_note, next_note, categories)
        if category not in categorized_notes:
            categorized_notes[category] = []
        categorized_notes[category].append(note)

    for category, notes in categorized_notes.items():
        print(f"\n{category}:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
