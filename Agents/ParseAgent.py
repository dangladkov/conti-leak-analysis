import os
import json
import pandas as pd
import re
import time
from tqdm.notebook import tqdm

folder_path = "Leak-files"
json_split_pattern = r'(?<=})\s*(?={)'

def ParseJson():
    messages_final = []
    start_time = time.time()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                messages_unstructured = f.read().strip()  # Reads unstructured json messages per file: {json msg}{json msg}...
                messages_structured = "[" + re.sub(json_split_pattern, ",",
                                                   messages_unstructured) + "]"  # Structures all json messages: [{json msg}, {json msg}...]
                try:
                    data = json.loads(messages_structured)  # Prepares data for tabulation
                    for msg in data:
                        msg["file"] = filename  # Optional filename
                        messages_final.append(msg)
                except json.JSONDecodeError as e:
                    print(f"Error parsing {filename}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(messages_final)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    end_time = time.time()
    print(f"Time taken to parse: {end_time - start_time}")

    # Count total number of characters in the 'body' column
    total_chars = df["body"].astype(str).str.len().sum()
    print("Total characters in 'body' column:", total_chars)

    return df.sort_values('ts')
print("Parser Agent activated.")
