import json
import argparse
import datetime
import os
import re # Import the regular expression module
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

CONTENT_DIR = "content"
OUTPUT_DIR = "cover_letters"

def main():
    parser = argparse.ArgumentParser(description="Generate a PDF cover letter from a template.")

    parser.add_argument("--company", required=True, help="The name of the company.")
    parser.add_argument("--body", required=True, help="Filename of the .txt file in the 'content' directory.")
    parser.add_argument("--manager", help="Optional: The name of the hiring manager.")
    parser.add_argument("--closing", default="Yours sincerely,", help="Optional: The closing phrase.")
    # New argument for keywords. 'nargs="*"' allows multiple keywords.
    parser.add_argument("--keywords", nargs='*', help="Optional list of keywords to bold in the text.")

    args = parser.parse_args()

    print("Starting cover letter generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        with open("my_info.json", "r") as f:
            my_info = json.load(f)
    except FileNotFoundError:
        print("Error: my_info.json not found.")
        return

    body_file_path = os.path.join(CONTENT_DIR, args.body)
    try:
        with open(body_file_path, "r") as f:
            body_text = f.read()
    except FileNotFoundError:
        print(f"Error: Body file not found at '{body_file_path}'")
        return

    # --- Keyword Highlighting Logic ---
    if args.keywords:
        print(f"Highlighting keywords: {args.keywords}")
        for keyword in args.keywords:
            # This regex ensures we match whole words only, case-insensitively
            body_text = re.sub(r'\b(' + re.escape(keyword) + r')\b', r'<strong>\1</strong>', body_text, flags=re.IGNORECASE)

    if args.manager:
        salutation = f"Dear {args.manager},"
    else:
        salutation = f"Dear {args.company} Hiring Team,"

    body_paragraphs = body_text.strip().split('\n\n')
    current_date = datetime.datetime.now().strftime('%B %d, %Y')

    template_data = {**my_info, "date": current_date, "recipient_name": args.manager, "company_name": args.company, "salutation": salutation, "body_paragraphs": body_paragraphs, "closing": args.closing}

    company_name_formatted = args.company.replace(" ", "")

    output_filename = f"CoverLetter-ReanFernandes_{company_name_formatted}.pdf"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("template.html")
    html_out = template.render(template_data)

    HTML(string=html_out, base_url='.').write_pdf(output_path)
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    main()