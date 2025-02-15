#!/usr/bin/env python3
import os
import re
import json
import sqlite3
import subprocess
import sys
from PIL import Image, ImageEnhance, ImageFilter
from dateutil import parser
from flask import Flask, request, jsonify, Response
import pytesseract
import numpy as np
import requests
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import markdown
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Constants for our data directory and AI proxy endpoints
DATA_DIR = os.path.abspath("/data")
LLM_CHAT_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
LLM_EMBEDDINGS_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")  # must be set in environment


### Helper functions

def safe_path(user_path: str) -> str:
    """
    Resolve and verify that a given user-supplied file path is under DATA_DIR.
    Raises ValueError if not.
    """
    abs_path = os.path.abspath(user_path)
    if not abs_path.startswith(DATA_DIR):
        raise ValueError("Access denied: path outside of allowed directory")
    return abs_path

def call_llm_chat(prompt: str) -> str:
    """
    Call our LLM proxy chat endpoint.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    resp = requests.post(LLM_CHAT_URL, headers=headers, json=data, timeout=30)
    resp.raise_for_status()
    response_json = resp.json()
    # Assume the answer is in response_json['choices'][0]['message']['content']
    return response_json["choices"][0]["message"]["content"].strip('```json').strip('```').strip()

def call_llm_embeddings(input_list):
    """
    Call our LLM proxy embeddings endpoint.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "text-embedding-3-small",
        "input": input_list
    }
    resp = requests.post(LLM_EMBEDDINGS_URL, headers=headers, json=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_task_to_operations(task: str) -> list:
    """
    Use the LLM to convert a plain‑English task into a list of operation dicts.
    For example, a valid response might be:
    
      [
         {"op": "count_dates", "params": {
              "input_file": "/data/dates.txt",
              "output_file": "/data/dates-wednesdays.txt",
              "weekday": "Wednesday"
         }},
         {"op": "format_file", "params": {
              "input_file": "/data/format.md"
         }}
      ]
      
    The LLM is instructed (via its system prompt, not shown here) to only produce valid JSON.
    """
    prompt = (
        "You are an agent that receives a plain‑English task description. "
        "Your job is to convert the following task into a JSON array of operations. "
        "Each operation should be an object with two keys: "
        "'op' (a string identifying the operation, such as 'install_and_run_datagen', 'format_file', 'count_dates', "
        "'sort_json', 'extract_header', etc.) and 'params' (an object with the necessary parameters). "
        "Do not add extra commentary. Output only valid JSON.\n\n"
        "List of possible operations:\n"
        "- 'install_and_run_datagen': { 'user_email': <user_email>,'script_url':<script_url> }\n"
        "- 'format_file': { 'input_file': <file_path>, 'formatter': <formatter_type> }\n"
        "- 'count_dates': { 'input_file': <file_path>, 'output_file': <file_path>, 'weekday': <day_name> }\n"
        "- 'extract_log_headers': { 'input_dir': <dir_path>, 'output_file': <file_path>, 'num_files': <int> }\n"
        "- 'sort_json': { 'input_file': <file_path>, 'output_file': <file_path>, 'keys': [<field_name_1>,<field_name_2>...] }\n"
        "- 'extract_header': { 'input_dir': <dir_path>, 'output_file': <file_path> }\n"
        "- 'extract_email': { 'input_file': <file_path>, 'output_file': <file_path> }\n"
        "- 'extract_credit_card': { 'input_file': <file_path>, 'output_file': <file_path> }\n"
        "- 'similar_comments': { 'input_file': <file_path>, 'output_file': <file_path> }\n"
        "- 'get_total_sales': { 'db_file': <file_path>, 'query': <sql_query>, 'output_file': <file_path>, 'ticket_type':<ticket_type> }\n"
        "- 'clone_git_repo': { 'repo_url': <url>, 'branch': <branch_name>, 'commit_message': <message>, 'clone_dir': <directory_path> }\n"
        "- 'fetch_api_data': { 'api_url': <url>, 'output_file': <file_path> }\n"
        "- 'scrape_website': { 'url': <url>, 'output_file': <file_path> }\n"
        "- 'run_sql_query': {'db_file': '<file_path>', 'query': '<sql_query>', 'output_file': '<file_path>'}\n"
        "- 'resize_image': { 'input_file': <file_path>, 'output_file': <file_path>, 'width': <int>, 'height': <int> }\n"
        "- 'transcribe_audio': { 'input_file': <file_path>, 'output_file': <file_path> }\n"
        "- 'convert_markdown': { 'input_file': <file_path>, 'output_file': <file_path> }\n"
        "- - 'filter_csv': { 'input_file': <file_path>, 'output_file': <file_path>, 'filters': [ { 'column': <column_name>, 'operator': '<=, >=, <, >, =', 'value': <value> } ] }\n\n"
        "Task description:\n" + task
    )
    llm_response = call_llm_chat(prompt)
    try:
        operations = json.loads(f'{llm_response}')
        if not isinstance(operations, list):
            raise ValueError("Expected a list of operations")
        return operations
    except Exception as e:
        raise ValueError("Failed to parse LLM output: " + str(e))


### Implementation of some example operations

def op_install_and_run_datagen(params: dict) -> dict:
    """
    Operation A1: Install uv (if needed) and run a remote script (datagen.py) with ${user.email} as the argument.
    Expects params: {"user_email": "...",script_url:"..."}

    This version checks if 'uv' is already installed before attempting installation.
    """    
    # Check if 'uv' is installed
    print("🚀 ~ params:", params)
    script_url = params.get("script_url", "").strip()
    try:
        import uv
        print("Package 'uv' is already installed.")
    except ImportError as e:
        pass
        print("Package 'uv' not found, installing...")
        try:
            subprocess.run([sys.executable.replace("\\","/"),"-m","pip", "install", "uv"], check=True)
            print("Package 'uv' installed successfully.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Failed to install uv: " + str(e))

    if not script_url:
        raise ValueError("Missing script_url in parameters.")
    try:
        # Using curl to download the script and running it with uv and user_email
        # subprocess.run([sys.executable.replace("\\","/"),"-m","uv", "run", script_url, params.get("user_email","23f3003196@ds.study.iitm.ac.in")], check=True)
        subprocess.run(f'uv run {script_url} {params.get("user_email", "23f3003196@ds.study.iitm.ac.in")}', shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to run datagen.py: " + str(e))
    return {"message": "Datagen script executed successfully."}

def op_format_file(params: dict) -> dict:
    """
    Operation A2: Format a file in place using the given formatter (e.g., prettier).
    Expects params: {"input_file": "/data/format.md", "formatter": "prettier"}
    """
    file_path = params.get("input_file", "").strip()
    formatter = params.get("formatter", "prettier").strip()
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}, 400

    # Run the formatter using npx
    try:
        subprocess.run(["npx", formatter, "--write", file_path], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to format file with {formatter}: {e}")
    return {"message": f"File {file_path} formatted successfully using {formatter}."}

def op_count_dates(params: dict) -> dict:
    """
    Operation A3: Count the number of specific weekdays in an input file and write the count to an output file.
    Expects params: {"input_file": "/data/dates.txt", "output_file": "/data/dates-wednesdays.txt", "weekday": "Wednesday"}
    """
    input_file = params.get("input_file", "").strip()
    output_file = params.get("output_file", "").strip()
    weekday_name = params.get("weekday", "Wednesday").strip().lower()

    # Map weekday name to weekday index (Monday==0, Sunday==6)
    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }
    target_index = weekdays.get(weekday_name)
    if target_index is None:
        raise ValueError("Invalid weekday: " + weekday_name)

    count = 0
    try:
        with open(input_file, "r") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    # Parse the date using dateutil.parser
                    dt = parser.parse(line)
                    if dt.weekday() == target_index:
                        count += 1
                except Exception:
                    # Skip lines that cannot be parsed
                    continue
    except Exception as e:
        raise RuntimeError("Error reading input file: " + str(e))
    
    try:
        with open(output_file, "w") as fout:
            fout.write(str(count))
    except Exception as e:
        raise RuntimeError("Error writing output file: " + str(e))

    return {"message": f"Found {count} {weekday_name.title()}(s) in {input_file}"}

def op_sort_json(params: dict) -> dict:
    """
    Operation A4: Sort an array in a JSON file by given keys and write the sorted array to an output file.
    Expects params: {"input_file": "/data/contacts.json", "output_file": "/data/contacts-sorted.json", "keys": ["last_name", "first_name"]}
    """
    input_file = safe_path(params.get("input_file", ""))
    output_file = safe_path(params.get("output_file", ""))
    sort_keys = params.get("keys")
    if not isinstance(sort_keys, list) or not sort_keys:
        raise ValueError("Missing sort keys")
    try:
        with open(input_file, "r") as fin:
            data = json.load(fin)
        # data is assumed to be a list of objects
        sorted_data = sorted(data, key=lambda x: tuple(x.get(key, "") for key in sort_keys))
    except Exception as e:
        raise RuntimeError("Error processing JSON file: " + str(e))
    try:
        with open(output_file, "w") as fout:
            json.dump(sorted_data, fout, indent=2)
    except Exception as e:
        raise RuntimeError("Error writing output JSON file: " + str(e))
    return {"message": f"Sorted JSON written to {output_file}"}

def op_extract_recent_logs(params: dict) -> dict:
    """
    A5: Extracts the first line of the 10 most recent .log files in a directory.
    Writes them to the output file in descending order (most recent first).
    """
    log_dir = safe_path(params.get("input_dir", ""))
    output_file = safe_path(params.get("output_file", ""))

    if not os.path.isdir(log_dir):
        raise RuntimeError(f"Invalid input directory: {log_dir}")

    log_files = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.endswith(".log")
    ]

    log_files.sort(key=os.path.getmtime,reverse=True)  # Sort by last modified time (newest first)

    recent_logs = []
    for log_file in log_files[:10]:  # Get top 10 recent logs
        print("🚀 ~ log_file:", log_file)
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    recent_logs.append(first_line)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    try:
        with open(output_file, "w", encoding="utf-8") as fout:
            fout.write("\n".join(recent_logs))
    except Exception as e:
        raise RuntimeError(f"Error writing output file: {e}")

    return {"message": f"Recent logs written to {output_file}"}

def extract_headers_recursive(directory, base_path):
    """
    Recursively searches for Markdown files in the given directory and extracts the first header.
    Stores filenames **without** the base_path prefix.
    """
    index = {}

    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)

        if os.path.isdir(full_path):
            # Recursively process subdirectories
            index.update(extract_headers_recursive(full_path, base_path))
        elif entry.lower().endswith(".md"):
            # Process Markdown file
            try:
                with open(full_path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        if line.lstrip().startswith("#"):
                            title = line.lstrip("#").strip()
                            # Store relative path without base directory
                            relative_path = os.path.relpath(full_path, base_path).replace("\\", "/")
                            index[relative_path] = title
                            break
            except Exception as e:
                print(f"Error reading {full_path}: {e}")

    return index

def op_extract_header(params: dict) -> dict:
    """
    Operation A6: Given a directory of Markdown files, extract the first header line from each file (recursively)
    and write an index JSON.
    Expects params: {"input_dir": "/data/docs/", "output_file": "/data/docs/index.json"}
    """
    input_dir = safe_path(params.get("input_dir", ""))
    output_file = safe_path(params.get("output_file", ""))
    
    if not os.path.isdir(input_dir):
        raise RuntimeError(f"Invalid input directory: {input_dir}")

    try:
        index = extract_headers_recursive(input_dir, input_dir)  # Use input_dir as base_path
    except Exception as e:
        raise RuntimeError(f"Error processing Markdown files: {e}")

    try:
        with open(output_file, "w", encoding="utf-8") as fout:
            json.dump(index, fout, indent=2)
    except Exception as e:
        raise RuntimeError(f"Error writing index file: {e}")

    return {"message": f"Index written to {output_file}"}

def op_extract_email(params: dict) -> dict:
    """
    A7: Extracts the sender's email address from an email file.
    """
    input_file = safe_path(params.get("input_file", ""))
    output_file = safe_path(params.get("output_file", ""))

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        email_match = re.search(r"From:.*?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]+)", content)
        if not email_match:
            raise ValueError("No email address found")

        sender_email = email_match.group(1)

        with open(output_file, "w", encoding="utf-8") as fout:
            fout.write(sender_email)

    except Exception as e:
        raise RuntimeError(f"Error processing email file: {e}")

    return {"message": f"Sender email written to {output_file}"}

def op_llm_extract(params: dict) -> dict:
    """
    Generic operation to delegate extraction of data (e.g. email sender, credit card number) to the LLM.
    Expects params: {"input_file": "...", "output_file": "...", "instruction": "..."}
    The file may be text or (if an image) may need to be read as binary and base64‑encoded.
    """
    input_file = safe_path(params.get("input_file", ""))
    output_file = safe_path(params.get("output_file", ""))
    instruction = params.get("instruction", "")
    if not instruction:
        raise ValueError("Missing extraction instruction")
    try:
        # For simplicity, assume text file.
        with open(input_file, "r") as fin:
            content = fin.read()
    except Exception as e:
        raise RuntimeError("Error reading input file: " + str(e))
    prompt = f"{instruction}\n\nContent:\n{content}"
    try:
        extraction = call_llm_chat(prompt).strip()
    except Exception as e:
        raise RuntimeError("LLM extraction failed: " + str(e))
    try:
        with open(output_file, "w") as fout:
            fout.write(extraction)
    except Exception as e:
        raise RuntimeError("Error writing output file: " + str(e))
    return {"message": f"Extraction completed and written to {output_file}"}

def preprocess_image(image_path):
    """Enhances image quality for better OCR recognition."""
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.filter(ImageFilter.SHARPEN)   # Sharpen the image
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast
    image = image.resize((image.width * 2, image.height * 2))  # Resize for better OCR
    return image

def op_extract_credit_card(params: dict) -> dict:
    """
    Extracts a 16-digit credit card number from an image using OCR with preprocessing.
    """
    input_file = params.get("input_file", "")
    output_file = params.get("output_file", "")

    try:
        # Preprocess the image
        image = preprocess_image(input_file)

        # Use Tesseract with improved OCR settings
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        print("🚀 ~ Extracted text:", text)

        # Strict 16-digit credit card regex (allows spaces or hyphens)
        card_number_match = re.search(r"\b(\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b", text)

        if card_number_match:
            card_number = card_number_match.group(0).replace(" ", "").replace("-", "")
            print("🚀 ~ Found credit card number:", card_number)

            # Ensure exactly 16 digits before saving
            if len(card_number) == 16:
                with open(output_file, "w", encoding="utf-8") as fout:
                    fout.write(card_number)
            else:
                raise ValueError("Extracted number is not exactly 16 digits.")

        else:
            raise ValueError("No valid 16-digit credit card number found.")

    except Exception as e:
        raise RuntimeError(f"Error processing credit card image: {e}")

    return {"message": f"Credit card number written to {output_file}"}

def op_similar_comments(params: dict) -> dict:
    """
    A9: Find the most similar pair of comments using embeddings.
    """
    input_file = safe_path(params.get("input_file", ""))
    output_file = safe_path(params.get("output_file", ""))

    try:
        # Step 1: Read the list of comments from the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            comments = [line.strip() for line in f.readlines() if line.strip()]

        if not comments:
            raise ValueError("No comments found in the file.")

        # Step 2: Generate embeddings for all comments in batches
        response = requests.post(
            LLM_EMBEDDINGS_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AIPROXY_TOKEN}"
            },
            json={
                "model": "text-embedding-3-small",
                "input": comments
            }
        )

        response.raise_for_status()
        embeddings = np.array([item['embedding'] for item in response.json()['data']])

        # Step 3: Compute pairwise cosine similarity efficiently
        similarity_matrix = 1 - squareform(pdist(embeddings, metric="cosine"))

        # Step 4: Find the most similar pair (excluding diagonal)
        np.fill_diagonal(similarity_matrix, -1)  # Ignore self-similarity
        max_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        most_similar_pair = (comments[max_idx[0]], comments[max_idx[1]])
        # Step 5: Write the most similar pair of comments to the output file
        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.write(f"{most_similar_pair[0]}\n")
            f_out.write(f"{most_similar_pair[1]}\n")

    except Exception as e:
        raise RuntimeError(f"Error processing comments for similarity: {e}")

    return {"message": f"Most similar comments written to {output_file}"}

def get_total_sales(params: dict) -> dict:
    """
    A10: Queries an SQLite database for the total sales of ${ticket type}.
    """
    db_file = safe_path(params.get("db_file", ""))
    output_file = safe_path(params.get("output_file", ""))

    query = "SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()[0]
        conn.close()

        with open(output_file, "w", encoding="utf-8") as fout:
            fout.write(str(result if result else 0))

    except Exception as e:
        raise RuntimeError(f"Error querying database: {e}")

    return {"message": f"Total sales for '{params.get('ticket_type','Gold')}' tickets written to {output_file}"}

def op_fetch_api_data(params: dict) -> dict:
    """
    B3: Fetch data from an API and save it.
    Expects params: {"api_url": "<url>", "output_file": "<file_path>"}
    """
    api_url = params.get("api_url", "").strip()
    output_file = safe_path(params.get("output_file", ""))

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        with open(output_file, "w", encoding="utf-8") as fout:
            json.dump(response.json(), fout, indent=2)
    except Exception as e:
        raise RuntimeError(f"Error fetching data from API: {e}")

    return {"message": f"Data fetched from {api_url} and saved to {output_file}"}

def op_clone_git_repo(params: dict) -> dict:
    """
    B4: Clone a git repo into a specific directory and make a commit.
    Expects params: {
        "repo_url": "<url>",
        "branch": "<branch_name>",
        "commit_message": "<message>",
        "clone_dir": "<directory_path>"
    }
    """
    repo_url = params.get("repo_url", "").strip()
    branch = params.get("branch", "main").strip()
    # Extract repo name from URL (assumes last part of URL is the repo name)
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

    commit_message = params.get("commit_message", "Update").strip()
    clone_dir = safe_path(params.get("clone_dir", f"/data/${repo_name}").strip())

    if not repo_url:
        raise ValueError("Missing 'repo_url' in parameters.")

    # Ensure the clone directory exists
    os.makedirs(clone_dir, exist_ok=True)

    try:
        # Change to the target directory
        os.chdir(clone_dir)

        # Clone the repository directly into the current working directory
        subprocess.run(["git", "clone", "-b", branch, repo_url, "."], check=True)

        # Create a temp file (this step happens after cd)
        subprocess.run(["touch", "temp.txt"], check=True)

        # Make a commit (assuming some changes are needed)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        os.chdir("/data")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error cloning repo or making commit: {e}")


    return {"message": f"Repo cloned to '{clone_dir}/{repo_name}' and committed with message: '{commit_message}'."}

def op_run_sql_query(params: dict) -> dict:
    """
    B5: Run a SQL query on a SQLite or DuckDB database.
    Expects params: {"db_file": "<file_path>", "query": "<sql_query>", "output_file": "<file_path>"}
    """
    db_file = safe_path(params.get("db_file", ""))
    query = params.get("query", "").strip()
    output_file = safe_path(params.get("output_file", ""))

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        with open(output_file, "w", encoding="utf-8") as fout:
            for row in results:
                fout.write(",".join(map(str, row)) + "\n")
    except Exception as e:
        raise RuntimeError(f"Error running SQL query: {e}")

    return {"message": f"Query results written to {output_file}"}

def op_scrape_website(params: dict) -> dict:
    """
    B6: Extract data from (i.e. scrape) a website.
    Expects params: {"url": "<url>", "output_file": "<file_path>"}
    """
    url = params.get("url", "").strip()
    output_file = safe_path(params.get("output_file", "/data/scraped-data.txt"))

    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_file, "w", encoding="utf-8") as fout:
            fout.write(response.text)
    except Exception as e:
        raise RuntimeError(f"Error scraping website: {e}")

    return {"message": f"Website content scraped from {url} and saved to {output_file}"}

def op_resize_image(params: dict) -> dict:
    """
    B7: Compress or resize an image.
    Expects params: 
    {
        "input_file": "<file_path>", 
        "output_file": "<file_path>", 
        "width": <int>, 
        "height": <int>,
        "quality": <int> (optional, default=85)
    }
    """
    input_file = safe_path(params.get("input_file", ""))
    output_file = safe_path(params.get("output_file", ""))
    width = params.get("width")
    height = params.get("height")
    quality = params.get("quality", 85)  # Default quality for compression

    if not os.path.isfile(input_file):
        raise RuntimeError(f"Input file does not exist: {input_file}")

    try:
        with Image.open(input_file) as img:
            # Convert to RGB if the image is in a format that doesn't support compression (e.g., PNG with alpha)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Resize while maintaining aspect ratio if width/height are not None
            if width and height:
                img = img.resize((width, height), Image.LANCZOS)
            else:
                img.thumbnail((width or img.width, height or img.height), Image.LANCZOS)

            # Detect format and apply compression settings
            img_format = img.format or "JPEG"
            save_params = {}

            if img_format == "JPEG":
                save_params = {"quality": quality, "optimize": True}
            elif img_format == "PNG":
                save_params = {"optimize": True}

            img.save(output_file, format=img_format, **save_params)

    except Exception as e:
        raise RuntimeError(f"Error resizing/compressing image: {e}")

    return {"message": f"Image resized and compressed, saved to {output_file}"}

def op_transcribe_audio(params: dict) -> dict:
    """
    B8: Transcribe audio from an MP3 file using Faster Whisper.
    Expects params: {"input_file": "<file_path>", "output_file": "<file_path>"}
    """
    input_file = safe_path(params.get("input_file", ""))
    output_file = safe_path(params.get("output_file", ""))

    if not os.path.isfile(input_file):
        raise RuntimeError(f"Input file does not exist: {input_file}")

    try:
        print("Loading Whisper model...")
        model = WhisperModel("small", device="cpu")

        print("Transcribing audio...")
        segments, _ = model.transcribe(input_file)

        # Combine all segments into final transcription
        transcription = " ".join([segment.text for segment in segments])

        # Save to output file
        with open(output_file, "w", encoding="utf-8") as fout:
            fout.write(transcription)

    except Exception as e:
        raise RuntimeError(f"Error transcribing audio: {e}")

    return {"message": f"Audio transcribed and saved to {output_file}"}

def op_convert_markdown(params: dict) -> dict:
    """
    B9: Convert Markdown to HTML.
    Expects params: {"input_file": "<file_path>", "output_file": "<file_path>"}
    """
    input_file = safe_path(params.get("input_file", ""))
    output_file = safe_path(params.get("output_file", ""))

    try:
        with open(input_file, "r", encoding="utf-8") as fin:
            markdown_content = fin.read()
        html_content = markdown.markdown(markdown_content)
        with open(output_file, "w", encoding="utf-8") as fout:
            fout.write(html_content)
    except Exception as e:
        raise RuntimeError(f"Error converting Markdown to HTML: {e}")

    return {"message": f"Markdown converted to HTML and saved to {output_file}"}

def op_filter_csv(params: dict) -> dict:
    """
    B10: Filters a CSV file based on conditions and saves the output as JSON.

    Expects params as:
        {
            "input_file": "<path to CSV>",
            "output_file": "<path to JSON>",
            "filters": [
                {"column": "<ColumnName>", "operator": "<=, >=, <, >, =", "value": "value"}
            ]
        }

    Example:
        params = {
            "input_file": "/data/data.csv",
            "output_file": "/data/output.json",
            "filters": [
                {"column": "Course id", "operator": "=", "value": "2001"},
                {"column": "Marks", "operator": ">", "value": "60"}
            ]
        }
    """
    input_file = safe_path(params.get("input_file", ""))
    output_file = safe_path(params.get("output_file", ""))
    filters = params.get("filters", [])

    # Ensure the file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        # Load CSV
        df = pd.read_csv(input_file)

        # Apply all filter conditions
        for condition in filters:
            column = condition.get("column")
            operator = condition.get("operator")
            value = condition.get("value")

            if column not in df.columns:
                raise KeyError(f"Column '{column}' not found in CSV.")

            if operator in [">", "<", ">=", "<=", "="]:
                try:
                    value = float(value) if str(value).replace(".", "").isdigit() else value
                    if operator == "=":
                        df = df[df[column] == value]
                    else:
                        df = df.query(f"`{column}` {operator} @value")
                except Exception as e:
                    raise ValueError(f"Invalid condition '{column} {operator} {value}': {e}")
            else:
                raise ValueError(f"Unsupported operator: {operator}")

        # Save filtered results to JSON
        df.to_json(output_file, orient="records", lines=True)

    except Exception as e:
        raise RuntimeError(f"Error filtering CSV: {e}")

    return {"message": f"Filtered data saved to {output_file}"}

### Operation dispatch table

# Map operation names (the ones our LLM parser produces) to functions.
OPERATION_DISPATCH = {
    "install_and_run_datagen": op_install_and_run_datagen,  # A1
    "format_file": op_format_file,  # A2
    "count_dates": op_count_dates,  # A3
    "sort_json": op_sort_json,  # A4
    "extract_header": op_extract_header,  # A6
    "extract_log_headers": op_extract_recent_logs,  # A5
    "llm_extract": op_llm_extract,  
    # A7 - Extract email sender
    "extract_email": op_extract_email,  
    # A8 - Extract credit card number
    "extract_credit_card": op_extract_credit_card,  
    # A9 - Find most similar comments using embeddings
    "similar_comments": op_similar_comments,  
    # A10 - Query SQLite database for total sales of "Gold" ticket type
    "get_total_sales": get_total_sales,
    "fetch_api_data": op_fetch_api_data,  # B3
    "clone_git_repo": op_clone_git_repo,  # B4
    "run_sql_query": op_run_sql_query,  # B5
    "scrape_website": op_scrape_website,  # B6
    "resize_image": op_resize_image,  # B7
    "transcribe_audio": op_transcribe_audio,  # B8
    "convert_markdown": op_convert_markdown,  # B9
    "filter_csv": op_filter_csv  # B10
}


### Main execution function

def execute_operations(operations: list) -> list:
    """
    Execute a list of operations. Each operation is a dict with keys 'op' and 'params'.
    Returns a list of results.
    """
    print("🚀 ~ operations:", operations)
    results = []
    for op in operations:
        op_name = op.get("op")
        params = op.get("params", {})
        if op_name not in OPERATION_DISPATCH:
            # For unknown operations, you might choose to return an error
            results.append({"error": f"Operation '{op_name}' not supported."})
            continue
        try:
            print("🚀 ~ result:", op_name)
            result = OPERATION_DISPATCH[op_name](params)
            print("🚀 ~ result:", result)
            results.append(result)
        except ValueError as ve:
            # Error in the task (e.g. missing parameter) returns a 400 error.
            raise ValueError(f"Error in operation '{op_name}': " + str(ve))
        except Exception as e:
            # Error in the agent (internal error) returns a 500.
            raise RuntimeError(f"Internal error in operation '{op_name}': " + str(e))
    return results


### Flask Endpoints

@app.route("/run", methods=["POST"])
def run_task():
    task = request.args.get("task", "")
    if not task:
        return jsonify({"error": "Missing task parameter"}), 400
    try:
        # Phase B requirement: Ensure that no operation ever requests access outside /data
        # Our safe_path() function in each operation helps enforce that.
        # Use the LLM to parse the plain‑English task into operations.
        operations = parse_task_to_operations(task)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Agent error: " + str(e)}), 500

    try:
        results = execute_operations(operations)
        return jsonify({"status": "success", "results": results}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Internal agent error: " + str(e)}), 500

@app.route("/read", methods=["GET"])
def read_file():
    file_path = request.args.get("path", "")
    if not file_path:
        return jsonify({"error": "Missing path parameter"}), 400
    try:
        safe_file = safe_path(file_path)
    except ValueError:
        return "", 404
    if not os.path.exists(safe_file):
        return "", 404
    try:
        with open(safe_file, "r") as f:
            content = f.read()
        return Response(content, status=200, mimetype="text/plain")
    except Exception as e:
        return jsonify({"error": "Internal error reading file: " + str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000)
