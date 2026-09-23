from pathlib import Path

import requests

from grupo_whatsapp.config import load_params


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        url,
        timeout=30,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if "text/html" in content_type:
        raise ValueError(
            "Expected a text file but SharePoint returned HTML. "
            "The URL may not be a direct download link."
        )

    output_path.write_bytes(response.content)

    print(f"Downloaded file to {output_path}")
    print(f"Content-Type: {content_type}")


def main() -> None:
    params = load_params()

    url = params["data"]["source_url"]
    output_path = Path(params["data"]["raw_file"])

    download_file(url, output_path)


if __name__ == "__main__":
    main()