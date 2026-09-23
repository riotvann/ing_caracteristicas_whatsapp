import re
from pathlib import Path

import pandas as pd

from grupo_whatsapp.config import PROJ_ROOT


INPUT_FILE = PROJ_ROOT / "data" / "raw" / "whatsapp.txt"
OUTPUT_FILE = PROJ_ROOT / "data" / "processed" / "whatsapp_messages.csv"


RECORD_PATTERN = re.compile(
    r"^"
    r"(\d{1,2}/\d{1,2}/\d{2,4}),\s*"
    r"(\d{1,2}:\d{2})[\s\u202f]*([AP]M)"
    r"\s*-\s*"
    r"(.*)$"
)

MESSAGE_CONTENT_PATTERN = re.compile(
    r"^([^:]+):\s*(.*)$"
)

URL_PATTERN = r"https?://[^\s]+"

MEDIA_PATTERN = (
    r"<(?:image|video|sticker|audio|document|gif) omitted>"
    r"|<album message>"
)

POKEMON_NAMES = [
    "Pikachu",
    "Charizard",
    "Bulbasaur",
    "Squirtle",
    "Jigglypuff",
    "Meowth",
    "Psyduck",
    "Snorlax",
    "Eevee",
    "Gengar",
    "Dragonite",
    "Mewtwo",
    "Lapras",
    "Vaporeon",
    "Jolteon",
    "Flareon",
    "Machamp",
    "Alakazam",
    "Gyarados",
    "Ditto",
    "Cubone",
    "Scyther",
    "Onix",
    "Arcanine",
    "Slowpoke",
    "Magikarp",
    "Abra",
    "Gastly",
    "Haunter",
    "Chansey",
]


def parse_whatsapp_txt(file_path: Path) -> list[dict]:
    messages = []
    current_message = None

    with file_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n\r")

            record_match = RECORD_PATTERN.match(line)

            if record_match:
                if current_message is not None:
                    messages.append(current_message)
                    current_message = None

                date, time, am_pm, body = record_match.groups()

                message_match = MESSAGE_CONTENT_PATTERN.match(body)

                if message_match:
                    sender, content = message_match.groups()

                    current_message = {
                        "date_raw": date,
                        "time_raw": f"{time} {am_pm}",
                        "sender": sender.strip(),
                        "content": content.strip(),
                    }

            elif current_message is not None:
                current_message["content"] += "\n" + line

    if current_message is not None:
        messages.append(current_message)

    return messages


def classify_content(content: str) -> str:
    content_lower = content.lower()

    media_types = {
        "<image omitted>": "image",
        "<video omitted>": "video",
        "<sticker omitted>": "sticker",
        "<audio omitted>": "audio",
        "<document omitted>": "document",
        "<gif omitted>": "gif",
        "<contact card omitted>": "contact",
        "<album message>": "album",
    }

    for marker, content_type in media_types.items():
        if marker in content_lower:
            return content_type

    if re.search(URL_PATTERN, content):
        return "link"

    return "text"


def anonymize_content(text: str, mapping: dict) -> str:
    if pd.isna(text):
        return text

    for real_name in sorted(mapping, key=len, reverse=True):
        text = re.sub(
            re.escape(real_name),
            mapping[real_name],
            text,
            flags=re.IGNORECASE,
        )

    return text


def process_chat(input_file: Path) -> pd.DataFrame:
    messages = parse_whatsapp_txt(input_file)

    df = pd.DataFrame(messages)

    if df.empty:
        raise ValueError("No WhatsApp messages were found in the input file.")

    # Datetime
    df["datetime"] = pd.to_datetime(
        df["date_raw"] + " " + df["time_raw"],
        format="%m/%d/%y %I:%M %p",
        errors="coerce",
    )

    # Content type
    df["content_type"] = df["content"].apply(classify_content)

    # Forwarded messages
    df["is_forwarded"] = df["content"].str.contains(
        r"\[Forwarded(?: many times)?\]",
        case=False,
        regex=True,
        na=False,
    )

    # URLs
    df["has_url"] = df["content"].str.contains(
        URL_PATTERN,
        regex=True,
        na=False,
    )

    df["url"] = df["content"].str.extract(
        f"({URL_PATTERN})",
        expand=False,
    )

    # Date/time features
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()

    # Message characteristics
    df["message_length"] = df["content"].str.len()
    df["word_count"] = df["content"].str.split().str.len()

    # Clean forwarded markers
    df["clean_content"] = (
        df["content"]
        .str.replace(
            r"\[Forwarded(?: many times)?\]",
            "",
            regex=True,
        )
        .str.strip()
    )

    # Media captions
    df["media_caption"] = (
        df["clean_content"]
        .str.replace(
            MEDIA_PATTERN,
            "",
            regex=True,
        )
        .str.strip()
    )

    df["has_caption"] = (
        df["content_type"].isin(
            [
                "image",
                "video",
                "audio",
                "document",
                "gif",
                "album",
            ]
        )
        & df["media_caption"].ne("")
    )

    # Text suitable for text/NLP analysis
    df["text"] = (
        df["clean_content"]
        .str.replace(
            MEDIA_PATTERN,
            "",
            regex=True,
        )
        .str.strip()
    )

    # Anonymize participants
    participants = sorted(df["sender"].dropna().unique())

    if len(participants) > len(POKEMON_NAMES):
        raise ValueError(
            f"There are {len(participants)} participants but only "
            f"{len(POKEMON_NAMES)} anonymous names."
        )

    participant_mapping = {
        participant: POKEMON_NAMES[i]
        for i, participant in enumerate(participants)
    }

    df["sender"] = df["sender"].map(participant_mapping)

    df["content"] = df["content"].apply(
        lambda value: anonymize_content(
            value,
            participant_mapping,
        )
    )

    df["text"] = df["text"].apply(
        lambda value: anonymize_content(
            value,
            participant_mapping,
        )
    )

    # Message ID
    df.insert(
        0,
        "message_id",
        range(1, len(df) + 1),
    )

    columns = [
        "message_id",
        "datetime",
        "date",
        "time",
        "year",
        "month",
        "day",
        "day_of_week",
        "hour",
        "sender",
        "content_type",
        "is_forwarded",
        "has_url",
        "url",
        "has_caption",
        "message_length",
        "word_count",
        "content",
        "text",
    ]

    return df[columns]


def validate_data(df: pd.DataFrame) -> None:
    invalid_datetimes = df["datetime"].isna().sum()
    missing_senders = df["sender"].isna().sum()
    missing_content = df["content"].isna().sum()
    duplicated_ids = df["message_id"].duplicated().sum()

    print(f"Messages: {len(df):,}")
    print(f"Participants: {df['sender'].nunique()}")
    print(f"Invalid datetimes: {invalid_datetimes}")
    print(f"Missing senders: {missing_senders}")
    print(f"Missing content: {missing_content}")
    print(f"Duplicated message IDs: {duplicated_ids}")

    if invalid_datetimes:
        raise ValueError(
            f"Found {invalid_datetimes} messages with invalid datetime."
        )

    if duplicated_ids:
        raise ValueError("Duplicated message IDs found.")


def main() -> None:
    print(f"Processing: {INPUT_FILE}")

    df = process_chat(INPUT_FILE)

    validate_data(df)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved {len(df):,} messages to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()