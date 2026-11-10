import ast
import pandas as pd
import re
from collections import defaultdict
from datetime import timedelta
from tqdm.notebook import tqdm
import time
import os
import textwrap
from langchain_ollama import OllamaLLM
import json
from msticpy.sectools.tilookup import TILookup
from msticpy.context.geoip import GeoLiteLookup
import msticpy as mp


###  Perpetrator profiling functions  ###


def build_user_profiles(df):
    profiles = defaultdict(lambda: {
        "username": "",
        "messages_sent": 0,
        "first_seen": None,
        "last_seen": None,
        "contacts": set(),
        "mentioned_emails": set(),
        "mentioned_wallets": set(),
        "mentioned_ips": set(),
    })

    progress = tqdm(total=len(df), desc="Building profiles")

    for i, row in df.iterrows():
        try:
            user = row["from"]
            ts = row["ts"]
            original_msg = row.get("body", "")
            translated_msg = row.get("translated_body", "")
            contact = row.get("to", "")

            profile = profiles[user]
            profile["username"] = user

            profile["messages_sent"] += 1
            profile["first_seen"] = min(profile["first_seen"], ts) if profile["first_seen"] else ts
            profile["last_seen"] = max(profile["last_seen"], ts) if profile["last_seen"] else ts
            profile["contacts"].add(contact)

            entities = extract_entities(original_msg)
            profile["mentioned_emails"].update(entities["emails"])
            profile["mentioned_wallets"].update(entities["wallets"])
            profile["mentioned_ips"].update(entities["ipv4"])

        except Exception as e:
            print(f"[ERROR] Row {i}: {e}")

        progress.update(1)

    progress.close()

    for p in profiles.values():
        for k in ["contacts", "mentioned_emails", "mentioned_wallets", "mentioned_ips"]:
            p[k] = list(p[k])

    return profiles

def safe_recursive_summary(user, texts, max_chars=3000):
    """Recursively summarize a list of LLM summaries focused on one user's concrete identifying details."""
    joined = "\n".join(texts)
    if len(joined) <= max_chars:
        prompt = f"""You are a cybercrime intelligence analyst.

Below are multiple partial finding reports derived from different message segments written by the same suspected ransomware actor nicknamed '{user}'. Your task is to combine these into one concise text containing ONLY concrete identifying intelligence about the user.

Your response should usually consist of TWO sentences, namely:
Sentence 1: Any self-reported or implied PERSONAL information such as name, aliases, time zone, nationality, location (not about someone else - specifically about this user)
Sentence 2: SHORT mention of their role in the group (if clearly obvious - ignore irrelevant banter, fluff or technical expertise), if the following format: "Potential role: [your estimation], [SHORT PHRASE as supporting evidence]"

Write a single paragraph following the template above, with no formatting or prompt-like phrases. Your output SHOULD be roughly equal in total length to the input summaries combined — DO NOT expand unnecessarily. Only include RELEVANT and FACTUAL FINDINGS, straight to the point. If you consider some personal information very important to the investigation into this user (like location reports and very personal info), you MUST keep it. If it doesn't fit into Sentence 1, you MUST make another sentence for it.

TEXT:
{joined}
"""
        print(f"---------------\nPrompt:\n{prompt}\n---------------")
        return llm.invoke(prompt)
    else:
        mid = len(texts) // 2
        left = safe_recursive_summary(user, texts[:mid], max_chars)
        right = safe_recursive_summary(user, texts[mid:], max_chars)
        return safe_recursive_summary(user, [left, right], max_chars)



class PortfolioBuilder:
    @staticmethod
    def chunk_text(convo_lines, max_chars):
        chunks = []
        current_chunk = []
        current_length = 0

        for line in convo_lines:
            line_length = len(line) + 5  # padding for newline etc.

            if current_length + line_length > max_chars:
                chunks.append(current_chunk)
                current_chunk = []
                current_length = 0

            current_chunk.append(line)
            current_length += line_length

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


    def build_user_prompt(user, text, previous_summary=""):
        return f"""You are a cybercrime intelligence analyst. Below are messages authored by a suspected member of a ransomware group nicknamed '{user}'. Write a concise findings report about this user.

Your response should consist of TWO sentences, namely:
Sentence 1: Any self-reported or implied PERSONAL information such as names, aliases, time zone, nationality, location (IMPORTANT NOTE: Sometimes the user forwards messages from other users, identified by "[timestamp] <user>" patterns in a message - DO NOT be fooled by this, those are messages from another user!)
Sentence 2: SHORT mention of their role in the group (if clearly obvious - ignore irrelevant banter, fluff or technical expertise), if the following format: "Potential role: [your estimation], [SHORT PHRASE as supporting evidence]"

Write a single paragraph following the template above, with no formatting or prompt-like phrases. Do not include opinions, summaries of intent, group dynamics, behavioral traits, or references to others. Avoid generalizations. Only output verifiable or directly stated details about the user. You can ONLY consider a name as self-reported if the user clearly states 'I am (name)...' or 'my name is...' or something along those lines - DO NOT include a name if that is not the case! Do not include formatting or prompt-style language, output just the report. DO NOT re-tell or summarize the messages or explain he what their technical expertise is - give me ONLY RELEVANT FACTUAL FINDINGS.

If nothing from these messages is of great importance to this investigation, output "None".

PREVIOUS REPORT (for context only):
{previous_summary or "None"}

MESSAGES:
{text}
"""

    
    @staticmethod
    def summarize_user_messages(user_messages_dict, max_total_chars=4096, prompt_chars=800, save_path="user_summaries.json"):
        import json, os

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                summaries = json.load(f)
            print(f"[Resume] Loaded {len(summaries)} already-summarized users.")
        else:
            summaries = {}

        max_chunk = max_total_chars - prompt_chars
        print(f"[PortfolioBuilder] Starting LLM summarization for {len(user_messages_dict)} users...")

        for i, (user, messages) in enumerate(user_messages_dict.items(), 1):
            if user in summaries:
                print(f"  🔁 Skipping already summarized user {user}")
                continue

            print(f"\n🧠 [{i}/{len(user_messages_dict)}] Summarizing messages for {user}...")

            formatted_messages = ["[*] " + msg.strip().replace("\n", " ") for msg in messages if msg.strip()]
            full_text = "\n".join(formatted_messages)

            if len(full_text) <= max_chunk:
                print(f"  → Sending 1 chunk ({len(full_text)} chars)")
                prompt = PortfolioBuilder.build_user_prompt(user, full_text)
                print(f"---------------\nPrompt:\n{prompt}\n--------------------")
                summary = llm.invoke(prompt)
                print(f"---------------\nGenerated Summary:\n{summary}\n--------------------")
            else:
                message_chunks = PortfolioBuilder.chunk_text(formatted_messages, max_chunk)
                chunks = ["\n".join(chunk) for chunk in message_chunks]

                print(f"  → Too long, splitting into {len(chunks)} chunks (~{max_chunk} chars each)")
                chunk_summaries = []
                previous_summary = ""
                for j, chunk in enumerate(chunks, 1):
                    print(f"    ↳ Chunk {j}/{len(chunks)}")
                    contextual_prompt = PortfolioBuilder.build_user_prompt(user, chunk, previous_summary)
                    print(f"---------------\nPrompt:\n{contextual_prompt}\n---------------")
                    response_chunk = llm.invoke(contextual_prompt)
                    print(f"---------------\nChunk Summary:\n{response_chunk}\n---------------")
                    previous_summary = response_chunk
                    chunk_summaries.append(response_chunk)

                summary = safe_recursive_summary(user, chunk_summaries)
                print(f"  → Final recursive summary:\n{summary}")

            summaries[user] = summary.strip()
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(summaries, f, ensure_ascii=False, indent=2)

        print("\n[PortfolioBuilder] All user message summaries completed.\n")
        return summaries


    
###  Victim analyzer functions  ###

    
class VictimAnalyzer:

    # Exhaustive victim-related regex pattern
    VICTIM_KEYWORDS_REGEX = re.compile(
        r'''(?ix)
        (
            # core ransomware/victim-related
            ransom(?:ware)?|
            hacked|
            leak(?:ed|age)?|
            publish(?:ed|ing)?|

            # victim identity / organizational info
            company|
            companies|
            firm|
            organization|
            business|
            corp(?:oration)?|
            department|
            employees?|
            website|
            site|

            # money and financial discussion
            price|
            amount|
            deal|
            discount|
            offer|
            demand|
            negotiate|
            afford|
            budget|
            paid|
            btc|
            bitcoin|
            usd|
            eur|

            # formatted monetary figures (only when standalone)
            (?:^|(?<=\s))
            (?:\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?|
            \$\s?\d+[kKmMbB]?|
            \d+\s?(k|kk|mil|m|million|b|billion))
            (?=\s|[.,!?)]|$)|

            # urgency and time (only when standalone)
            (?:^|(?<=\s))
            (?:deadline|urgenc(?:y|ies))
            (?=\s|[.,!?)]|$)
        )
        ''',
        re.IGNORECASE | re.VERBOSE
    )



    @staticmethod
    def detect_victim_leads(df):
        leads = []
        for _, row in df.iterrows():
            message = str(row.get("translated_body", ""))
            match = VictimAnalyzer.VICTIM_KEYWORDS_REGEX.search(message)
            if match:
                row_data = row.to_dict()
                row_data["trigger_keyword"] = match.group(0)
                leads.append(row_data)
        return pd.DataFrame(leads)


    @staticmethod
    def extract_context_windows(df, leads_df, window_size=15, max_chars=2800):
        grouped = df.groupby("chat_pair_id")
        context_chunks = []

        leads_grouped = leads_df.groupby("chat_pair_id")

        for convo_id, lead_group in tqdm(leads_grouped, desc="Extracting context windows"):
            convo_df = grouped.get_group(convo_id).reset_index(drop=True)
            convo_df["char_len"] = convo_df["translated_body"].fillna("").astype(str).apply(len)

            # List of rows that triggered the regex in this conversation
            lead_indexes = lead_group["id"].tolist()

            # Slide a window across the conversation to include all lead messages
            start = 0
            while start < len(convo_df):
                end = min(start + window_size * 2 + 1, len(convo_df))
                chunk_df = convo_df.iloc[start:end]

                # Check if this chunk includes any lead messages
                chunk_ids = chunk_df["id"].tolist()
                if not any(lead_id in chunk_ids for lead_id in lead_indexes):
                    start += window_size  # skip if it doesn't cover any lead
                    continue

                # Build message block with char limit
                chunk_lines = []
                total_len = 0
                for _, row in chunk_df.iterrows():
                    if not isinstance(row['translated_body'], str):
                        continue
                    clean_body = str(row['translated_body']).replace("\n", " ").strip()
                    line = f"[*] {row['from']}: {clean_body}"

                    if total_len + len(line) > max_chars:
                        break
                        
                    chunk_lines.append(line)
                    total_len += len(line) + 2  # for newlines

                chunk_text = "\n".join(chunk_lines)

                context_chunks.append({
                    "chat_pair_id": convo_id,
                    "lead_message_id": chunk_df.iloc[0]["id"],  # anchor for file naming
                    "chunk_text": chunk_text
                })

                # Move the window forward — either all or partial
                start = end

        return context_chunks



    @staticmethod
    def summarize_victim_case(chunk_text, llm_fn):
        prompt = f"""You are a cybercrime intelligence analyst. Below is a conversation involving potential ransomware victim negotiations.

Extract the following details as a Python-style list:
["company_name", "domain", "ransom_amount", "ransom_currency", "urgency_hours", "attack_type", "data_leaked", "attacker_tone", "notes"]

Possibly ambiguous fields meanings:
- urgency_hours: hours or days until deadline, if mentioned
- data_leaked: "true" if publication has occurred or was threatened, "false" if explicitly stated otherwise
- attacker_tone: e.g. threatening, cooperative, mocking (only if the discussion is victim-related, put "n/a" otherwise)
- notes: brief free-form detail if anything unusual or insightful is observed (only if the discussion is victim-related, put "n/a" otherwise)

Use the exact order above. For any unknown or missing value, insert "n/a" as a string. Do NOT include markdown, code formatting, code blocks, or language tags like ```python. Output only a raw Python-style list on a single line.

IMPORTANT: If you are reasonably sure that this conversation does NOT involve a victim or is NOT related to victim-related discussion, return:
["n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]

Conversation:
{chunk_text}
"""
        print(f"---------------\nPrompt:\n{prompt}\n--------------------")
        return llm_fn(prompt)


    @staticmethod
    def analyze_victims(df, llm_fn, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        print("[VictimAnalyzer] Detecting potential victim-related messages...")
        leads_df = VictimAnalyzer.detect_victim_leads(df)
        leads_df.to_csv(os.path.join(output_dir, "victim_leads.csv"), index=False)
        print(f"[VictimAnalyzer] {len(leads_df)} lead messages detected.")

        context_chunks = VictimAnalyzer.extract_context_windows(df, leads_df)

        results = []
        
        # Track processed message IDs
        existing_ids = set()
        existing_csv = os.path.join(output_dir, "victim_profiles.csv")
        if os.path.exists(existing_csv):
            existing_ids = set(pd.read_csv(existing_csv)["lead_message_id"].astype(str))
        
        # Filter only chunks with new lead_message_ids
        work_chunks = [chunk for chunk in context_chunks if str(chunk["lead_message_id"]) not in existing_ids]

        # Proper progress bar only over the chunks we're actually processing
        for i, chunk in enumerate(tqdm(work_chunks, desc="Analyzing victim leads"), 1):
            if str(chunk["lead_message_id"]) in existing_ids:
                continue

            print(f"\n📍 [{i}/{len(work_chunks)}] Analyzing chat pair {chunk['chat_pair_id']}...")
            
            raw_summary = VictimAnalyzer.summarize_victim_case(chunk["chunk_text"], llm_fn)
            raw_summary.strip()
            print(f"---------------\nGenerated Summary:\n{raw_summary}\n--------------------")
            try:
                parsed = ast.literal_eval(raw_summary.strip())
                print(f"parsed: {parsed}")
                if not isinstance(parsed, list) or len(parsed) != 9:
                    raise ValueError("Invalid list format")
            except Exception as e:
                print(f"[!] Parsing failed for message {chunk['lead_message_id']}: {e}")
                parsed = ["n/a"] * 9  # fallback

            result = {
                "chat_pair_id": chunk["chat_pair_id"],
                "lead_message_id": chunk["lead_message_id"],
                "conversation_context": chunk["chunk_text"].strip() if any(x != "n/a" for x in parsed) else "n/a",
                "company_name": parsed[0],
                "domain": parsed[1],
                "ransom_amount": parsed[2],
                "ransom_currency": parsed[3],
                "urgency_hours": parsed[4],
                "attack_type": parsed[5],
                "data_leaked": parsed[6],
                "attacker_tone": parsed[7],
                "notes": parsed[8]
            }

            results.append(result)
            # Save this result incrementally
            row_df = pd.DataFrame([result])
            row_df.to_csv(existing_csv, mode="a", index=False, header=not os.path.exists(existing_csv))
            print(f"[✔] Saved result for lead {chunk['lead_message_id']}", flush=True)

        return results



###  IOC analysis functions  ###


def extract_ioc_table(df, ioc_type: str, pattern: re.Pattern, label: str, output_dir: str):
    output_path = os.path.join(output_dir, f"{label}_table.csv")
    rows = []

    progress = tqdm(df.iterrows(), total=len(df), desc=f"Extracting {label}s")

    for _, row in progress:
        matches = pattern.findall(str(row.get("body", "")))
        for match in matches:
            ioc = match if isinstance(match, str) else match[0]
            rows.append({
                "ioc": ioc,
                "type": ioc_type,
                "from": row["from"],
                "to": row["to"],
                "timestamp": row["ts"],
                "message_id": row["id"],
                "message": row["body"],
                "translated_message": row.get("translated_body", "")
            })

    df_ioc = pd.DataFrame(rows)
    df_ioc.to_csv(output_path, index=False)
    tqdm.write(f"Saved to: {output_path}\n")
    time.sleep(0.3)


def generate_ioc_tables(df, output_dir="ioc-csvs"):
    os.makedirs(output_dir, exist_ok=True)

    ioc_types = {
        "wallets": "wallet",
        "ipv4": "ip",
        "urls": "url",
        "emails": "email"
    }

    progress = tqdm(total=len(df))

    for key, label in ioc_types.items():
        rows = []

        progress.reset(total=len(df))
        progress.set_description(f"Extracting {label}s")

        for _, row in df.iterrows():
            matches = regex_patterns[key].findall(str(row.get("body", "")))
            for match in matches:
                ioc = match if isinstance(match, str) else match[0]
                rows.append({
                    "ioc": ioc,
                    "type": label,
                    "from": row["from"],
                    "to": row["to"],
                    "timestamp": row["ts"],
                    "message_id": row["id"],
                    "message": row["body"],
                    "translated_message": row.get("translated_body", "")
                })
            progress.update(1)

        df_ioc = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, f"{label}_table.csv")
        df_ioc.to_csv(out_path, index=False)
        tqdm.write(label.capitalize() + f"s saved to: {out_path}")
        time.sleep(0.3)
    
    progress.set_description(f"Extraction complete")
    progress.close()

    
def enrich_ip_table_with_msticpy(ip_table_path, output_path):
    print("[+] Loading IP table...")
    df = pd.read_csv(ip_table_path)
    df = df[df['type'] == 'ip'].copy()
    df.rename(columns={"ioc": "ip"}, inplace=True)

    print("[+] Performing Threat Intel lookups...")
    ti_results = []
    for ip in tqdm(df["ip"], desc="TI Lookup"):
        try:
            result_df = ti_lookup.lookup_iocs([ip])
            if not result_df.empty:
                ti_results.append(result_df.iloc[0])
            else:
                ti_results.append({})
        except Exception as e:
            print(f"[!] Error with TI lookup for {ip}: {e}")
            ti_results.append({})

    ti_df = pd.DataFrame(ti_results)
    ti_df.index = df.index  # align rows

    print("[+] Performing GeoIP lookups...")
    geo_results = []
    for ip in tqdm(df["ip"], desc="GeoIP Lookup"):
        try:
            loc_result, _ = geo_lookup.lookup_ip(ip_address=ip)
            geo_results.append(loc_result[0] if loc_result else {})
        except Exception as e:
            print(f"[!] Error with GeoIP lookup for {ip}: {e}")
            geo_results.append({})

    geo_df = pd.DataFrame(geo_results)
    geo_df.index = df.index  # align rows

    print("[+] Merging enrichment results...")
    enriched = pd.concat([df, ti_df, geo_df], axis=1)

    print(f"[+] Saving enriched IP table to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    enriched.to_csv(output_path, index=False)
    print("✅ Done.")

    
def attach_enriched_ips_to_profiles(profiles, ip_csv_path="ioc-csvs/ip_table_enriched.csv"):
    import pandas as pd

    ip_df = pd.read_csv(ip_csv_path)

    # Normalize column names
    if "ip" not in ip_df.columns and "ioc" in ip_df.columns:
        ip_df.rename(columns={"ioc": "ip"}, inplace=True)

    # Drop duplicate IPs
    ip_df = ip_df.drop_duplicates(subset="ip")

    # Set IP as index and build lookup
    ip_lookup = ip_df.set_index("ip").to_dict(orient="index")

    for username, profile in profiles.items():
        enriched_list = []
        new_ip_entries = {}

        for ip in profile.get("mentioned_ips", []):
            ip_data = ip_lookup.get(ip)
            ip_entry = {}

            if ip_data:
                country = ip_data.get("country")
                if country and isinstance(country, str) and "nan" not in country.lower():
                    try:
                        # Parse stringified dicts
                        country_dict = ast.literal_eval(country)
                        continent_dict = ast.literal_eval(ip_data.get("continent", "{}"))
                        location_dict = ast.literal_eval(ip_data.get("location", "{}"))

                        ip_entry.update({
                            "country_name": country_dict.get("names", {}).get("en"),
                            "continent_name": continent_dict.get("names", {}).get("en"),
                            "provider": ip_data.get("Provider"),
                            "location": location_dict,
                            "status": ip_data.get("Status"),
                            "severity": ip_data.get("Severity")
                        })
                    except Exception as e:
                        print(f"[WARN] Failed to parse enrichment for IP {ip}: {e}")

            new_ip_entries[ip] = ip_entry

        # Replace original mentioned_ips with enriched versions
        profile["mentioned_ips"] = new_ip_entries

    return profiles


def normalize_username(raw):
    """
    Strip domains like .onion from addresses like 'user@xyz.onion'.
    """
    return raw.split("@")[0] if "@" in raw else raw

    
def classify_user_ips_with_llm(profiles, df, max_context_chars=2500):
    def normalize_username(name):
        return name.split("@")[0] if "@" in name else name

    for username, profile in tqdm(profiles.items(), desc="LLM IP classification"):
        user_msgs = df[df["from"].apply(normalize_username) == username]
        mentioned_ips = profile.get("mentioned_ips", {})

        for ip, ip_data in tqdm(mentioned_ips.items(), desc=f"→ IPs for {username}", leave=False):
            user_rows = df[df["from"].apply(normalize_username) == username][["translated_body"]].reset_index()
            msg_tuples = list(zip(user_rows["index"], user_rows["translated_body"]))

            context_lines = []
            for idx, text in msg_tuples:
                if ip in str(text):
                    # Get a range from 3 messages before to 3 messages after (including current)
                    for offset in range(-3, 4):  # [-3, -2, -1, 0, +1, +2, +3]
                        neighbor_idx = idx + offset
                        if neighbor_idx in df.index:
                            line = df.loc[neighbor_idx, "translated_body"]
                            context_lines.append(line)

            context_lines = [l for l in context_lines if isinstance(l, str) and l.strip()]
            seen = set()
            context_lines = [l for l in context_lines if not (l in seen or seen.add(l))]
            formatted = [f"[*] {l.replace(chr(10), ' ')}" for l in context_lines]
            chat_context = "\n".join(formatted)[:max_context_chars] if formatted else "No context found."

            ip_metadata = "\n".join([f"{k}: {v}" for k, v in ip_data.items() if k != "agent_analysis"])

            prompt = textwrap.dedent(f"""You are a cybercrime intelligence analyst. You are given message(s) from a user that mention an IP address. Your job is to determine the most appropriate label for how this IP relates to the user. Consider the IP metadata provided. Think carefully about whether the user owns the IP, operates malware or infrastructure on it, merely mentions it, or shows curiosity.

Your output must be a 3-part Python-style list in the form:
["LABEL", CONFIDENCE_SCORE (0-100), "REASON"]

- LABEL must be a short machine-friendly tag like: owns_ip, malware_host, resale, curious, planning_to_use, etc.
- CONFIDENCE_SCORE is how certain you are (from 0 to 100)
- REASON should be brief and based only on the user’s words.

IMPORTANT: Do not include markdown formatting, code blocks, or language tags like ```python. Output only a raw Python-style list on a single line.

IP: {ip}
IP METADATA:
{ip_metadata}

Messages mentioning IP:
{chat_context}
""")

            try:
                print(f"---------------\nPrompt:\n{prompt}\n--------------------")
                response = llm.invoke(prompt).strip()
                print(f"---------------\nResponse:\n{response}\n--------------------")
                parsed = ast.literal_eval(response)
                print(f"parsed: {parsed}")
                label, score, reason = parsed if isinstance(parsed, list) and len(parsed) == 3 else ("unknown", 0, "LLM parsing failed")
            except Exception as e:
                label, score, reason = "error", 0, f"LLM or parsing error: {e}"

            profiles[username]["mentioned_ips"][ip]["agent_analysis"] = {
                "label": label,
                "confidence": score,
                "reason": reason
            }

    return profiles


    
    
###  Main stuff  ###


def normalize_username(name):
    if isinstance(name, str) and "@" in name:
        return name.split("@")[0].strip().lower()
    return name.lower().strip() if isinstance(name, str) else name

def split_into_conversations(df, time_gap_minutes=30):
    df = df.sort_values("ts").copy()

    # New: Add stable from-to pair ID (regardless of time)
    df["chat_pair_id"] = df.apply(lambda row: tuple(sorted([row["from"], row["to"]])), axis=1)

    # Existing logic: Time-sensitive conversation ID
    df["conversation_id"] = 0
    conv_id = 0
    last_time = None
    last_pair = None

    for i, row in df.iterrows():
        current_pair = row["chat_pair_id"]
        if last_time is None or row["ts"] - last_time > timedelta(minutes=time_gap_minutes) or current_pair != last_pair:
            conv_id += 1
        df.at[i, "conversation_id"] = conv_id
        last_time = row["ts"]
        last_pair = current_pair

    return df

def extract_entities(text):
    entities = {}
    if not isinstance(text, str):
        return {key: [] for key in regex_patterns}
    for key, pattern in regex_patterns.items():
        entities[key] = pattern.findall(text)
    return entities

# Regex patterns
regex_patterns = {
    "emails": re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b', re.IGNORECASE),
    "domains": re.compile(r'\b(?:[a-zA-Z0-9-]+\.)+(com|net|ru|org|onion|xyz|io|biz|info|co|gov|us|uk|de|jp|cn)\b'),
    "ipv4": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    "wallets": re.compile(r'\b(?:' + '|'.join([
        r'bc1[a-z0-9]{39,59}',                            # Bitcoin Bech32 (bc1)
        r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}',               # Bitcoin legacy and P2SH
        r'0x[a-fA-F0-9]{40}',                             # Ethereum
        r'4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}',              # Monero
        r'X[1-9A-HJ-NP-Za-km-z]{33}',                     # Dash
    ]) + r')\b', re.IGNORECASE),
    "urls": re.compile(r'https?://[^\s]+'),
    "usernames": re.compile(r'@[\w\d_-]{3,}', re.IGNORECASE)
}

DATA_PATH = "translated_output.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["ts"])
df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
df = df.sort_values("ts").reset_index(drop=True)
df["from"] = df["from"].apply(normalize_username)
df["to"] = df["to"].apply(normalize_username)

llm = OllamaLLM(model="gemma3:27b", temperature=0.3)

mp.settings.refresh_config()

# Initialize MSTICPy lookups
ti_lookup = TILookup(providers=["VirusTotal"])
geo_lookup = GeoLiteLookup()

print(f"Analyzer Agent activated. Model used: {llm.model}")
