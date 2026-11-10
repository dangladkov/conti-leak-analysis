from Agents.ParseAgent import *

from langchain.prompts import PromptTemplate
import string
import re
from collections import Counter
from langchain_ollama import OllamaLLM
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from queue import Queue
import threading
import time
import os
import time
import queue
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm

# Shared metrics for logging translation stats
metrics = {
    "gibberish_attempts": 0,
    "gibberish_messages": set(),  # holds unique indexes
    "timeout_errors": 0
}

pd.set_option("display.max_colwidth", 0)
pd.set_option("display.width", 0)

llm = OllamaLLM(model="gemma3:12b", temperature=0.3)

KNOWN_TRANSLATIONS_PATH = "known_translations.json"

def load_known_translations():
    if os.path.exists(KNOWN_TRANSLATIONS_PATH):
        with open(KNOWN_TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_known_translation(phrase, translation):
    data = load_known_translations()
    data[phrase] = translation
    with open(KNOWN_TRANSLATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def normalize_msg(msg):
    if not isinstance(msg, str):
        return ""
    msg = re.sub(r'\s*\n\s*', ' ', msg.strip())  # Remove newlines and collapse spaces
    msg = msg.lower()
    msg = msg.rstrip(string.punctuation + "…")
    msg = re.sub(r'[!?.…]+$', '', msg)
    return msg.strip()

def extract_frequent_phrases(df, column, top_n):
    normalized = df[column].dropna().apply(normalize_msg)
    normalized = normalized[normalized != ""]
    normalized = normalized[~normalized.apply(is_skip_message)]  # Exclude skipped messages
    counter = Counter(normalized)
    return dict(counter.most_common(top_n))

def build_dynamic_translation_dict(phrases_dict, progress):
    translations = {}
    saved_translations = load_known_translations()

    for phrase in phrases_dict:
        if phrase in saved_translations:
            translations[phrase] = saved_translations[phrase]
        else:
            try:
                translated = llm.invoke(f"Translate the following Russian or transliterated Russian message to English. It's from a ransomware group's private chat on Jabber ('жаба'). Output ONLY the translated message, in one string without newlines, and nothing else, don't hallucinate additional information or commentary. Here is the message: {phrase.strip()}").strip()
                translated = re.sub(r'\s+', ' ', translated)
                lines = list(dict.fromkeys(translated.split()))
                translated = ' '.join(lines).strip()
                translations[phrase] = translated
                save_known_translation(phrase, translated)
            except Exception as e:
                translations[phrase] = f"Error: {e}"
        progress.update(1)
    return translations


def clean_message(msg):
        if not isinstance(msg, str):
            return ""
        return re.sub(r'\s*\n\s*', ' ', msg.strip())

def is_gibberish(original, translated, grace_ratio=2.5):
    if not isinstance(translated, str) or not isinstance(original, str):
        return True, "non-string input"

    original_len = len(original.strip())
    translated_len = len(translated.strip())

    # Rule 1: Hallucination on ultra-short inputs
    if original_len < 15 and translated_len > 40:
        return True, "short input → long output"

    # Rule 2: Output too long
    if original_len > 0 and (translated_len / original_len) > grace_ratio:
        return True, "output too long"

    # Rule 3: Cyrillic characters still present
    if re.search(r'[а-яА-ЯёЁ]', translated):
        return True, "Cyrillic in output"

    # Rule 4: Output too short for meaningful input
    if original_len > 6 and translated_len / original_len < 0.18:
        return True, "output too short"

    return False, None
    
def is_skip_message(msg):
    if pd.isna(msg) or not isinstance(msg, str):
        return True
    msg = msg.strip()
    return (
        msg == "" or
        bool(re.fullmatch(r'https?://\S+|www\.\S+', msg)) or
        bool(re.fullmatch(r'\d{1,2}:\d{2}(:\d{2})?', msg)) or
        bool(re.fullmatch(r'[\d\s\W_]+', msg))  # just digits/punctuation/whitespace
    )

def is_known_translation(msg, known_translations):
    return normalize_msg(msg) in known_translations

def get_known_translation(msg, known_translations):
    return known_translations.get(normalize_msg(msg), msg)

def safe_invoke(prompt):
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model="gemma3:12b")  # Recreate inside process
    return llm.invoke(prompt)

def run_with_timeout(prompt, timeout=100):
    with multiprocessing.Pool(processes=1) as pool:
        result = pool.apply_async(safe_invoke, (prompt,))
        try:
            return result.get(timeout)
        except multiprocessing.TimeoutError:
            pool.terminate()
            raise TimeoutError(f"Translation timed out after {timeout} seconds.")

def translate_single(index, message, all_messages):
    conversation = ""
    context_flow = []

    # Collect up to 2 previous messages
    for offset in range(2, 0, -1):
        prev_index = index - offset
        if prev_index in all_messages:
            prev_msg = clean_message(all_messages[prev_index])
            if len(prev_msg) <= 150:
                context_flow.append(prev_msg)

    # Add current message (to be translated)
    context_flow.append(message.strip())

    # Collect up to 2 next messages
    for offset in range(1, 3):
        next_index = index + offset
        if next_index in all_messages:
            next_msg = clean_message(all_messages[next_index])
            if len(next_msg) <= 150:
                context_flow.append(next_msg)

    # Only build additional note if context exists
    additional_note = ""
    if len(context_flow) > 1:
        conversation = []
        position = 1
        for msg in context_flow:
            prefix = "=> " if msg.strip() == message.strip() else "   "
            conversation.append(f"{position}. {prefix}{msg}")
            position += 1
            
    additional_note = ("\n\n[CONTEXT] The following messages are from the surrounding conversation. They are provided to help you understand tone or meaning. DO NOT translate them. Only translate the line marked with =>.\n\n" + "\n".join(conversation))
    
    prompt = ("You are a precise Russian-to-English translator.\n\n"
"Translate ONLY the marked message. Output ONLY the English translation.\n"
"Do not include language labels, formatting, or commentary. Do NOT translate the surrounding context.\n"
"Important:\n"
"- If the message includes source code, scripts, or commands, leave those sections unchanged.\n"
"- If the message is a chat log (e.g., includes timestamps like [HH:MM:SS], usernames, or tags like <name>), preserve all formatting and only translate the Russian text.\n"
"- Do not translate names, usernames, timestamps, or structural markers.\n"
"- If the message is gibberish or untranslatable, repeat it using Latin transliteration — do NOT use Cyrillic.\n\n"
f"TARGET MESSAGE:\n=> {message}"
)

    if additional_note and len(message) <= 1000:
        prompt += additional_note
        
    result_clean = ""
    attempt = 1
    while True:
        try:
            if attempt == 6:
                tqdm.write(f"[DEBUG] Message {index} taking too long, timeout extended")
            
            tqdm.write(f"[DEBUG] Message {index} prompted")
            if attempt >= 6:
                timeout = 60
            else:
                timeout = 45

            response_raw = run_with_timeout(prompt, timeout)
                
            result_clean = response_raw.strip()

            global gibberish_attempts
            # Reject gibberish outputs
            is_bad, reason = is_gibberish(message, result_clean)
            if is_bad:
                metrics["gibberish_attempts"] += 1
                metrics["gibberish_messages"].add(index)
                print(f"[DEBUG] Message {index}. Original: {message} -> Translated: {result_clean}")
                raise ValueError(f"Suspected gibberish: {reason}")

            # Only log success after passing checks
            if attempt == 1:
                tqdm.write(f"[DEBUG] Message {index} translated: {message}")
            else:
                tqdm.write(f"[DEBUG] Message {index} translated (in {attempt} attempts!)")

            return (index, result_clean)
        
        except (TimeoutError, ValueError) as e:
            if isinstance(e, TimeoutError):
                metrics["timeout_errors"] += 1
            elif isinstance(e, ValueError) and "Suspected gibberish" in str(e):
                metrics["gibberish_attempts"] += 1

            print(f"[DEBUG] Message {index} attempt {attempt}: {e} 🚨 Message: {message[:150]}")
            attempt += 1

            if attempt >= 12:
                tqdm.write(f"[DEBUG] Message {index} took too long, using latest translation: {result_clean}")
                return (index, result_clean)
            
        except Exception as e:
            print(f"[ERROR] Index {index}: {e}")
            return (index, f"Error: {e}")


    
### MAIN FUNCTION ###
    
def TranslateMessages(df, n, max_workers, to_process_known, save_path):
    import time
    import queue
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    print(f"Started translating {n} messages.")
    start_time = time.time()

    # Complete random sampling
    sample_df = df.sample(n=min(n, len(df))).copy()

    
    sample_df.index.name = "id"
    sample_df["translated_body"] = ""

    write_header = not os.path.exists(save_path)
    translated_map = {}

    # Load existing translations
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path, index_col="id")
        if "translated_body" in existing_df.columns:
            
            translated_map = {
                idx: val for idx, val in zip(existing_df.index, existing_df["translated_body"])
                if isinstance(val, str) and val.strip() != ""
            }
            
            tqdm.write(f"Resuming from existing file with {len(translated_map)} translations.")
        else:
            translated_map = {}
            tqdm.write(f"[INFO] Existing file found at {save_path}, but it has no 'translated_body' column. Starting fresh.")
    else:
        translated_map = {}
        tqdm.write(f"Resuming from existing file with {len(translated_map)} translations.")

    KNOWN_TRANSLATIONS = load_known_translations()
    KNOWN_TRANSLATIONS[""] = ""

    if to_process_known:
        frequent_phrases = extract_frequent_phrases(df, "body", 100)
        shared_progress = tqdm(total=len(frequent_phrases), desc="Frequent messages translation", unit="message")
        dynamic_dict = build_dynamic_translation_dict(frequent_phrases, shared_progress)
        shared_progress.set_description("Translations complete")
        time.sleep(1.0)
    else:
        shared_progress = tqdm(total=69, desc="Frequent messages translation", unit="message")  # TESTING

    shared_progress.set_description("Preprocessing messages")
    time.sleep(0.5)

    messages = list(sample_df["body"].items())
    shared_progress.reset(total=len(messages))

    tasks = []
    
    # for reporting
    is_skipped = 0
    is_known = 0
    
    for i, msg in messages:
        if i in translated_map:
            sample_df.at[i, "translated_body"] = translated_map[i]
        elif is_skip_message(msg):
            sample_df.at[i, "translated_body"] = msg
            translated_map[i] = msg
            tasks.append((i, None))  # Mark as "write now"
            is_skipped += 1
        elif is_known_translation(msg, KNOWN_TRANSLATIONS):
            translated = get_known_translation(msg, KNOWN_TRANSLATIONS)
            sample_df.at[i, "translated_body"] = translated
            translated_map[i] = translated
            tasks.append((i, None))  # Mark as "write now"
            is_known += 1
        else:
            cleaned = clean_message(msg)
            tasks.append((i, cleaned))
        shared_progress.update(1)

    print(f"Messages skipped: {is_skipped}, messages known-translated: {is_known}")
    shared_progress.set_description("Main translation")
    shared_progress.reset(total=len(tasks))

    write_queue = queue.Queue()
    stop_token = object()

    def writer_thread():
        file_exists = os.path.exists(save_path)
        with open(save_path, mode="a", newline="", encoding="utf-8") as f:
            write_header_local = not file_exists
            while True:
                item = write_queue.get()
                if item is stop_token:
                    break
                idx, row = item
                row.to_csv(f, header=write_header_local, index=True)
                write_header_local = False
                f.flush()  # Force flush internal buffer
                os.fsync(f.fileno())  # Force OS-level disk write
                write_queue.task_done()

    writer = threading.Thread(target=writer_thread)
    writer.start()

    all_messages_dict = dict(sample_df["body"])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(translate_single, i, msg, all_messages_dict): i
            for i, msg in tasks if msg is not None
        }

        for i, msg in tasks:
            if msg is None:  # Already known/skip
                write_queue.put((i, sample_df.loc[[i]]))
                shared_progress.update(1)

        for future in as_completed(futures):
            idx = futures[future]
            translated = future.result()[1]
            sample_df.at[idx, "translated_body"] = translated
            translated_map[idx] = translated
            write_queue.put((idx, sample_df.loc[[idx]]))
            shared_progress.update(1)

    write_queue.put(stop_token)
    writer.join()

    shared_progress.set_description("Done")
    sample_df["ts"] = pd.to_datetime(sample_df["ts"], errors="coerce")
    sample_df = sample_df.sort_values(by="ts")
    sample_df.index.name = "id"
    sample_df.to_csv(save_path, index=True)

    elapsed = time.time() - start_time
    print(f"Finished translating {n} messages in {elapsed:.2f} seconds.")
    
    print(f"Total gibberish translation attempts: {metrics['gibberish_attempts']}")
    print(f"Unique messages that triggered gibberish: {len(metrics['gibberish_messages'])}")
    print(f"Total TimeoutErrors: {metrics['timeout_errors']}")
    return sample_df


def FixGibberishTranslations(max_workers, main_csv_path="translated_output.csv", temp_csv_path="gibberish_subset.csv"):
    import pandas as pd
    from tqdm.notebook import tqdm

    # Step 1: Load full dataset
    df = pd.read_csv(main_csv_path, index_col="id")
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.sort_values(by="ts")

    tqdm.write(f"[INFO] Loaded {len(df)} messages.")

    # Load completed translations from the existing temporary file
    translated_subset = pd.read_csv(temp_csv_path, index_col="id")
    translated_subset = translated_subset[
        translated_subset["translated_body"].apply(lambda x: isinstance(x, str) and x.strip() != "")
    ]

    # Step 5: Overwrite the gibberish rows in main CSV
    for idx, row in translated_subset.iterrows():
        df.at[idx, "translated_body"] = row["translated_body"]

    # Step 6: Save final updated dataset
    df.to_csv(main_csv_path, index=True)
    tqdm.write(f"[DONE] Replaced {len(translated_subset)} gibberish rows in {main_csv_path}.")

    return df



print(f"Translator Agent activated. Model used: {llm.model}")
