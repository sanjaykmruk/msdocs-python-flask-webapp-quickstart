from typing import List
from pathlib import Path

def generate_md(product_metadata: List[dict], save_md=False, location="products_recs.md"):
    md_template = """| Product Title | Image |\n| --- | --- |"""

    for article in product_metadata:
        title = article["title"]
        article_image = article["imageUrl"]
        md_template += f"\n| {title} | ![text]({article_image}) |"

    if save_md:
        write_md(md_template, location=location)

    return md_template


def write_md(text, location="product_recs.md"):
    file_path = Path(location)
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        f.write(text)
        f.close()
