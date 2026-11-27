"""
loader.py - File Loader Module
================================
What this file does:
    Contains ALL file loading functions in ONE file.
    Each loader reads a specific file type (PDF, TXT, DOCX, Excel, CSV)
    and returns the text in a standard format.
    The load_file() function automatically picks the right loader
    based on the file extension.

Input:  A file path (string)
Output: A list of dictionaries, each with keys: "text", "location", "source"
        Example: {"text": "some text...", "location": "Page 3", "source": "paper1.pdf"}
"""

import os
import fitz  # PyMuPDF - the main PDF reader
from pypdf import PdfReader  # PyPDF - the fallback PDF reader
from docx import Document  # python-docx - for Word files
import pandas as pd  # pandas - for Excel and CSV files

from config import (
    LINES_PER_GROUP_TXT,
    PARAGRAPHS_PER_GROUP_DOCX,
    ROWS_PER_GROUP_TABLE,
    SUPPORTED_EXTENSIONS,
)


# ============================================================
# FUNCTION 1: load_pdf_with_pymupdf
# ============================================================
# What it does: Opens a PDF file using PyMuPDF (fitz) and extracts text
#               from each page. Returns one dictionary per page.
# Parameters:
#   file_path (str): The full path to the PDF file
# Returns:
#   A list of dictionaries with text, location, and source
def load_pdf_with_pymupdf(file_path):
    # Get just the file name (e.g., "paper1.pdf") from the full path
    file_name = os.path.basename(file_path)
    list_of_entries = []

    # Open the PDF with PyMuPDF
    pdf_document = fitz.open(file_path)
    total_pages = len(pdf_document)
    print(f"  [PyMuPDF] Opened '{file_name}' with {total_pages} pages.")

    # Loop through each page (page numbers start at 0 in fitz, but we label from 1)
    for page_number in range(total_pages):
        page = pdf_document[page_number]
        page_text = page.get_text()  # Extract all text from this page

        # Clean up the text: remove extra whitespace
        page_text = page_text.strip()

        # Skip pages that have no text (e.g., image-only pages)
        if page_text == "":
            print(f"  [PyMuPDF] Skipping page {page_number + 1} - no text found.")
            continue

        # Create the entry dictionary for this page
        entry = {
            "text": page_text,
            "location": f"Page {page_number + 1}",  # Human-readable page number
            "source": file_name,
        }
        list_of_entries.append(entry)

    # Close the PDF document
    pdf_document.close()
    print(f"  [PyMuPDF] Extracted text from {len(list_of_entries)} pages.")
    return list_of_entries


# ============================================================
# FUNCTION 2: load_pdf_with_pypdf
# ============================================================
# What it does: Opens a PDF file using PyPDF (fallback loader) and extracts
#               text from each page. Same output format as PyMuPDF loader.
# Parameters:
#   file_path (str): The full path to the PDF file
# Returns:
#   A list of dictionaries with text, location, and source
def load_pdf_with_pypdf(file_path):
    file_name = os.path.basename(file_path)
    list_of_entries = []

    # Open the PDF with PyPDF
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    print(f"  [PyPDF] Opened '{file_name}' with {total_pages} pages.")

    # Loop through each page
    for page_number in range(total_pages):
        page = reader.pages[page_number]
        page_text = page.extract_text()

        # extract_text() can return None if no text is found
        if page_text is None:
            page_text = ""

        page_text = page_text.strip()

        # Skip empty pages
        if page_text == "":
            print(f"  [PyPDF] Skipping page {page_number + 1} - no text found.")
            continue

        entry = {
            "text": page_text,
            "location": f"Page {page_number + 1}",
            "source": file_name,
        }
        list_of_entries.append(entry)

    print(f"  [PyPDF] Extracted text from {len(list_of_entries)} pages.")
    return list_of_entries


# ============================================================
# FUNCTION 3: load_pdf
# ============================================================
# What it does: Tries to load a PDF with PyMuPDF first. If that fails
#               or returns no text, it falls back to PyPDF.
# Parameters:
#   file_path (str): The full path to the PDF file
# Returns:
#   A list of dictionaries with text, location, and source
def load_pdf(file_path):
    file_name = os.path.basename(file_path)
    print(f"Loading PDF: {file_name}")

    # Try PyMuPDF first (it's usually better at extracting text)
    try:
        entries = load_pdf_with_pymupdf(file_path)
        if len(entries) > 0:
            print(f"  Successfully loaded with PyMuPDF.")
            return entries
        else:
            print(f"  PyMuPDF returned no text, trying PyPDF fallback...")
    except Exception as error:
        print(f"  PyMuPDF failed with error: {error}")
        print(f"  Trying PyPDF fallback...")

    # Fallback to PyPDF
    try:
        entries = load_pdf_with_pypdf(file_path)
        if len(entries) > 0:
            print(f"  Successfully loaded with PyPDF (fallback).")
            return entries
        else:
            print(f"  PyPDF also returned no text. The PDF might be image-only.")
            return []
    except Exception as error:
        print(f"  PyPDF fallback also failed with error: {error}")
        return []


# ============================================================
# FUNCTION 4: load_txt
# ============================================================
# What it does: Reads a plain text file, splits it into lines, and groups
#               every LINES_PER_GROUP_TXT lines into one entry.
#               Tries UTF-8 encoding first, then falls back to Latin-1.
# Parameters:
#   file_path (str): The full path to the TXT file
# Returns:
#   A list of dictionaries with text, location, and source
def load_txt(file_path):
    file_name = os.path.basename(file_path)
    print(f"Loading TXT: {file_name}")
    file_content = ""

    # Try reading with UTF-8 encoding first (most common)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
        print(f"  Read with UTF-8 encoding.")
    except UnicodeDecodeError:
        # If UTF-8 fails, try Latin-1 (which can read almost any file)
        try:
            with open(file_path, "r", encoding="latin-1") as file:
                file_content = file.read()
            print(f"  UTF-8 failed, read with Latin-1 encoding.")
        except Exception as error:
            print(f"  Error reading file: {error}")
            return []

    # Split the content into individual lines
    all_lines = file_content.split("\n")
    total_lines = len(all_lines)
    print(f"  Total lines: {total_lines}")

    list_of_entries = []
    group_size = LINES_PER_GROUP_TXT

    # Group lines together (e.g., lines 1-40, 41-80, etc.)
    for start_index in range(0, total_lines, group_size):
        end_index = start_index + group_size
        # Make sure we don't go past the end of the file
        if end_index > total_lines:
            end_index = total_lines

        # Get the lines for this group
        group_lines = all_lines[start_index:end_index]
        group_text = "\n".join(group_lines)
        group_text = group_text.strip()

        # Skip empty groups
        if group_text == "":
            print(f"  Skipping lines {start_index + 1}-{end_index} - empty text.")
            continue

        # Line numbers start at 1 for the user (not 0)
        location_label = f"Lines {start_index + 1}-{end_index}"

        entry = {
            "text": group_text,
            "location": location_label,
            "source": file_name,
        }
        list_of_entries.append(entry)

    print(f"  Created {len(list_of_entries)} entries from {total_lines} lines.")
    return list_of_entries


# ============================================================
# FUNCTION 5: load_docx
# ============================================================
# What it does: Reads a Word (.docx) file using python-docx, extracts
#               paragraphs and tables, and groups paragraphs into entries.
#
# NOTE: Word files do NOT store real page numbers. The page layout depends
#       on fonts, margins, printer settings, etc. So we use paragraph
#       ranges as location labels instead (e.g., "Paragraphs 1-10").
#
# Parameters:
#   file_path (str): The full path to the DOCX file
# Returns:
#   A list of dictionaries with text, location, and source
def load_docx(file_path):
    file_name = os.path.basename(file_path)
    print(f"Loading DOCX: {file_name}")

    try:
        doc = Document(file_path)
    except Exception as error:
        print(f"  Error opening DOCX file: {error}")
        return []

    # --- Step 1: Extract paragraphs ---
    # Collect all non-empty paragraphs
    all_paragraphs = []
    for paragraph in doc.paragraphs:
        paragraph_text = paragraph.text.strip()
        if paragraph_text != "":
            all_paragraphs.append(paragraph_text)

    total_paragraphs = len(all_paragraphs)
    print(f"  Found {total_paragraphs} non-empty paragraphs.")

    list_of_entries = []
    group_size = PARAGRAPHS_PER_GROUP_DOCX

    # Group paragraphs together
    for start_index in range(0, total_paragraphs, group_size):
        end_index = start_index + group_size
        if end_index > total_paragraphs:
            end_index = total_paragraphs

        # Get the paragraphs for this group
        group_paragraphs = all_paragraphs[start_index:end_index]
        group_text = "\n\n".join(group_paragraphs)
        group_text = group_text.strip()

        # Skip if the group is empty
        if group_text == "":
            print(f"  Skipping paragraphs {start_index + 1}-{end_index} - empty text.")
            continue

        location_label = f"Paragraphs {start_index + 1}-{end_index}"

        entry = {
            "text": group_text,
            "location": location_label,
            "source": file_name,
        }
        list_of_entries.append(entry)

    # --- Step 2: Extract tables ---
    # Word documents can also contain tables; we read those too
    table_number = 0
    for table in doc.tables:
        table_number = table_number + 1
        table_rows_text = []

        # Loop through each row in the table
        for row in table.rows:
            # Get the text from each cell in the row, joined with " | "
            cell_texts = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                cell_texts.append(cell_text)

            # Join all cell texts with " | " to make a readable row
            row_text = " | ".join(cell_texts)
            table_rows_text.append(row_text)

        # Combine all rows into one text block
        table_text = "\n".join(table_rows_text)
        table_text = table_text.strip()

        # Skip empty tables
        if table_text == "":
            print(f"  Skipping Table {table_number} - empty.")
            continue

        entry = {
            "text": table_text,
            "location": f"Table {table_number}",
            "source": file_name,
        }
        list_of_entries.append(entry)

    print(f"  Created {len(list_of_entries)} entries ({total_paragraphs} paragraphs + {table_number} tables).")
    return list_of_entries


# ============================================================
# FUNCTION 6: load_csv
# ============================================================
# What it does: Reads a CSV file with pandas, converts each row into a
#               text string (ColumnName: value | ...), and groups every
#               ROWS_PER_GROUP_TABLE rows into one entry.
#               Tries UTF-8 encoding first, then Latin-1.
# Parameters:
#   file_path (str): The full path to the CSV file
# Returns:
#   A list of dictionaries with text, location, and source
def load_csv(file_path):
    file_name = os.path.basename(file_path)
    print(f"Loading CSV: {file_name}")

    # Try reading with UTF-8 first, then Latin-1
    dataframe = None
    try:
        dataframe = pd.read_csv(file_path, encoding="utf-8")
        print(f"  Read with UTF-8 encoding.")
    except UnicodeDecodeError:
        try:
            dataframe = pd.read_csv(file_path, encoding="latin-1")
            print(f"  UTF-8 failed, read with Latin-1 encoding.")
        except Exception as error:
            print(f"  Error reading CSV file: {error}")
            return []
    except Exception as error:
        print(f"  Error reading CSV file: {error}")
        return []

    total_rows = len(dataframe)
    column_names = list(dataframe.columns)
    print(f"  Total rows: {total_rows}, Columns: {column_names}")

    # Convert each row into a text line like "Col1: val1 | Col2: val2 | ..."
    all_row_texts = []
    for row_index in range(total_rows):
        parts = []
        for col_name in column_names:
            # Get the value; replace NaN/missing with empty string
            value = dataframe.iloc[row_index][col_name]
            if pd.isna(value):
                value = ""
            else:
                value = str(value)
            parts.append(f"{col_name}: {value}")

        # Join all column:value pairs with " | "
        row_text = " | ".join(parts)
        all_row_texts.append(row_text)

    # Group rows together
    list_of_entries = []
    group_size = ROWS_PER_GROUP_TABLE

    for start_index in range(0, total_rows, group_size):
        end_index = start_index + group_size
        if end_index > total_rows:
            end_index = total_rows

        group_texts = all_row_texts[start_index:end_index]
        group_text = "\n".join(group_texts)
        group_text = group_text.strip()

        # Skip empty groups
        if group_text == "":
            print(f"  Skipping rows {start_index + 1}-{end_index} - empty text.")
            continue

        # Row numbers start at 1 for the user
        location_label = f"Rows {start_index + 1}-{end_index}"

        entry = {
            "text": group_text,
            "location": location_label,
            "source": file_name,
        }
        list_of_entries.append(entry)

    print(f"  Created {len(list_of_entries)} entries from {total_rows} rows.")
    return list_of_entries


# ============================================================
# FUNCTION 7: load_excel
# ============================================================
# What it does: Reads ALL sheets of an .xlsx file with pandas, converts
#               rows to text (same format as CSV), and groups every
#               ROWS_PER_GROUP_TABLE rows. Location includes sheet name.
# Parameters:
#   file_path (str): The full path to the Excel (.xlsx) file
# Returns:
#   A list of dictionaries with text, location, and source
def load_excel(file_path):
    file_name = os.path.basename(file_path)
    print(f"Loading Excel: {file_name}")

    try:
        # sheet_name=None reads ALL sheets into a dictionary
        # engine="openpyxl" is needed for .xlsx files
        all_sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
    except Exception as error:
        print(f"  Error reading Excel file: {error}")
        return []

    sheet_names = list(all_sheets.keys())
    print(f"  Found {len(sheet_names)} sheets: {sheet_names}")

    list_of_entries = []

    # Process each sheet separately
    for sheet_name in sheet_names:
        dataframe = all_sheets[sheet_name]
        total_rows = len(dataframe)

        # Skip empty sheets
        if total_rows == 0:
            print(f"  Skipping sheet '{sheet_name}' - empty.")
            continue

        column_names = list(dataframe.columns)
        print(f"  Sheet '{sheet_name}': {total_rows} rows, Columns: {column_names}")

        # Convert each row into a text line (same format as CSV)
        all_row_texts = []
        for row_index in range(total_rows):
            parts = []
            for col_name in column_names:
                value = dataframe.iloc[row_index][col_name]
                if pd.isna(value):
                    value = ""
                else:
                    value = str(value)
                parts.append(f"{col_name}: {value}")

            row_text = " | ".join(parts)
            all_row_texts.append(row_text)

        # Group rows together
        group_size = ROWS_PER_GROUP_TABLE

        for start_index in range(0, total_rows, group_size):
            end_index = start_index + group_size
            if end_index > total_rows:
                end_index = total_rows

            group_texts = all_row_texts[start_index:end_index]
            group_text = "\n".join(group_texts)
            group_text = group_text.strip()

            if group_text == "":
                print(f"  Skipping sheet '{sheet_name}', rows {start_index + 1}-{end_index} - empty.")
                continue

            # Include the sheet name in the location label
            location_label = f"Sheet: {sheet_name}, Rows {start_index + 1}-{end_index}"

            entry = {
                "text": group_text,
                "location": location_label,
                "source": file_name,
            }
            list_of_entries.append(entry)

    print(f"  Created {len(list_of_entries)} entries total from all sheets.")
    return list_of_entries


# ============================================================
# FUNCTION 8: load_file (THE ROUTER)
# ============================================================
# What it does: Looks at the file extension and calls the correct loader.
#               This is the ONLY function you need to call from outside.
# Parameters:
#   file_path (str): The full path to any supported file
# Returns:
#   A list of dictionaries with text, location, and source.
#   Returns an empty list if the file type is not supported.
def load_file(file_path):
    file_name = os.path.basename(file_path)

    # Get the file extension in lowercase, without the dot
    # For example: "paper1.PDF" -> "pdf"
    if "." in file_name:
        extension = file_name.rsplit(".", 1)[1].lower()
    else:
        extension = ""

    print(f"\n--- Loading file: {file_name} (type: .{extension}) ---")

    # Route to the correct loader based on extension
    if extension == "pdf":
        entries = load_pdf(file_path)
    elif extension == "txt":
        entries = load_txt(file_path)
    elif extension == "docx":
        entries = load_docx(file_path)
    elif extension == "xlsx":
        entries = load_excel(file_path)
    elif extension == "csv":
        entries = load_csv(file_path)
    else:
        print(f"  Unsupported file type: .{extension}")
        print(f"  Supported types are: {SUPPORTED_EXTENSIONS}")
        return []

    print(f"--- Finished loading {file_name}: {len(entries)} entries ---\n")
    return entries


# ============================================================
# FUNCTION 9: load_all_files_from_folder
# ============================================================
# What it does: Loops over all files in a folder, calls load_file for
#               each supported file, and returns one combined list.
# Parameters:
#   folder_path (str): The path to the folder containing files
# Returns:
#   A combined list of all entries from all supported files
def load_all_files_from_folder(folder_path):
    print(f"\n=== Loading all files from folder: {folder_path} ===")

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"  Folder does not exist: {folder_path}")
        return []

    # Get all file names in the folder
    all_file_names = os.listdir(folder_path)
    print(f"  Found {len(all_file_names)} files in folder.")

    combined_entries = []

    for file_name in all_file_names:
        # Build the full path
        file_path = os.path.join(folder_path, file_name)

        # Skip if it's a directory, not a file
        if not os.path.isfile(file_path):
            continue

        # Get the extension to check if it's supported
        if "." in file_name:
            extension = file_name.rsplit(".", 1)[1].lower()
        else:
            extension = ""

        if extension not in SUPPORTED_EXTENSIONS:
            print(f"  Skipping unsupported file: {file_name}")
            continue

        # Load this file and add its entries to the combined list
        entries = load_file(file_path)
        for entry in entries:
            combined_entries.append(entry)

    print(f"\n=== Total entries from all files: {len(combined_entries)} ===")
    return combined_entries


# ============================================================
# Test block: Run this file alone to test all loaders
# Usage: python loader.py
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LOADER MODULE TEST")
    print("=" * 60)

    # You can test with any files you have. Here we test the router
    # with whatever files exist in the data/uploads folder.
    import os

    test_folder = "data/uploads"

    if os.path.isdir(test_folder):
        all_files = os.listdir(test_folder)
        if len(all_files) > 0:
            print(f"\nFound files in {test_folder}: {all_files}")
            for file_name in all_files:
                file_path = os.path.join(test_folder, file_name)
                if os.path.isfile(file_path):
                    entries = load_file(file_path)
                    print(f"\n  Results for {file_name}:")
                    print(f"  Total entries: {len(entries)}")
                    # Print first 2 entries (first 200 chars each)
                    for i in range(min(2, len(entries))):
                        text_preview = entries[i]["text"][:200]
                        print(f"  Entry {i+1}:")
                        print(f"    Location: {entries[i]['location']}")
                        print(f"    Source: {entries[i]['source']}")
                        print(f"    Text: {text_preview}...")
        else:
            print(f"\nNo files found in {test_folder}.")
            print("To test, place some PDF, TXT, DOCX, XLSX, or CSV files there.")
    else:
        print(f"\nFolder {test_folder} does not exist.")
        print("Creating it now...")
        os.makedirs(test_folder, exist_ok=True)
        print("Place some files in data/uploads/ and run this test again.")

    # Quick test with a sample to show the router works
    print("\n--- Router test with a fake path ---")
    result = load_file("test.unsupported")
    print(f"Unsupported file result: {result}")
    print("\nLoader module test complete!")
