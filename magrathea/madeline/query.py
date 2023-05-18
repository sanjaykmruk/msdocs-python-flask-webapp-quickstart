from pluralizer import Pluralizer

pluralizer = Pluralizer()


def replace_first_last(string):
    if len(string) > 0 and string[0] == "|" and string[-1] == "|":
        string = string[1:-1]
    return string


def parse_gpt_response(text_input):
    """
    Example response:
    Category | Description

    Shoes | Classic ivory patent leather pumps with a 2-inch kitten heel; the perfect balance of sophistication and comfort for the special day.

    Dress | Simple, yet elegant A-Line dress in periwinkle blue; the light fabric and subtle lace detailing would look gorgeous against the backdrop of the Almafi Coast.

    Jewelry | Delicate gold teardrop earrings encrusted with small crystals; perfect for adding a touch of glamour to the outdoor wedding.
    """
    products = {}  # collect products
    # descriptions = []
    for line in text_input.split("\n"):
        line = replace_first_last(line)
        # GPT to sometimes split table with \t and sometimes |
        if "\t" in line:
            separator_in_response = "\t"
        else:
            separator_in_response = "|"

        if (
            (separator_in_response in line)
            and (f"Category{separator_in_response}Description" not in line.replace(" ", ""))
            and (line[0] != "-")
        ):
            category, text = line.split(separator_in_response)
            category = category.lower().strip()

            if not pluralizer.isPlural(category):
                category = pluralizer.plural(category)  # pluralize category

            text = text.replace(separator_in_response, " ")  # replace separator_in_response with space
            products[category] = text
            # descriptions.append(line)

    return products


product_desc_template = """Occasion: 
{chat_history}.
{product_search}.

Task:
Write up to 5 verbose womenswear product descriptions related to the occasion above.

Context:
These items should go together to form a complete outfit.
Only answer in tabular form, | separated, with two columns: category, description"""


def parse_chroma_resposes(chroma_responses):
    unique_id = []
    collated = []
    for result in chroma_responses:
        metadata = result["metadatas"][0]
        if metadata.get("article_id") not in unique_id:
            unique_id.append(metadata.get("article_id"))
            collated += metadata

    return collated
