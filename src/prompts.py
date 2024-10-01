generateCategoriesPrompt = """Below are notes I have written on a certain topic. Provide a list of sub topics which I can use to categorise these notes
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
    - any fluff/cringe/commentary/hype"""
    
splitPrompt = """
    Rules:
    - Split a note if and only if BOTH of these conditions are met:
        - The note contains meaningfully distinct sub-parts which clearly do not all belong under the same category
        - The resulting split notes make sense in isolation & do not depend on each other for context
    - Splits must only occur on newline characters
    - Do not split a note just because it has dividers/sub-sections
    - Do not split in the middle of a line of text
    - The resulting split pieces must add up exactly to the original note
        - i.e. do not add or remove ANY text from the original note, or re-order the text
"""