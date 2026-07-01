import datetime
import os
import traceback

import pandas as pd
import requests

EXCEL_FILE_NAME = "Uploaded Title -Year 2024.xlsx"
EXCEL_FILE = f"{os.path.dirname(__file__)}/{EXCEL_FILE_NAME}"

DOC_DIR_NAME = "2024"
DOC_DIR = f"{os.path.dirname(__file__)}/../data/docs/{DOC_DIR_NAME}"


print(f"{datetime.datetime.now().isoformat()} | Training | Start of the script")


df = pd.read_excel(EXCEL_FILE)

print(f"{datetime.datetime.now().isoformat()} | Training | Total books to train: {len(df['Print ISBN'].tolist())}")

token = "2gbVAVNw1Cug7KRJOTNQ9vETZ9WIe20nfPDOt0oB"

for isbn in df["Print ISBN"].tolist():
    print(f"{datetime.datetime.now().isoformat()} | Training | ISBN: {isbn} | Geting product details...")

    try:

        product = requests.get(f"https://api.packt.com/api/v2/products/{isbn}?token={token}")
        product.raise_for_status()
        book_id = product.json()["product_id"]

        print(
            f"{datetime.datetime.now().isoformat()} | Training | ISBN: {isbn}"
            f" | BOOK_ID: {book_id} | Downloading book..."
        )

        epub = requests.get(f"https://api.packt.com/api/v2/products/{isbn}/epub?token={token}")
        epub.raise_for_status()
        with open(f"{DOC_DIR}/{book_id}.epub", "wb") as file:
            file.write(epub.content)

        print(
            f"{datetime.datetime.now().isoformat()} | Training | ISBN: {isbn}" f" | BOOK_ID: {book_id} | Book stored!"
        )
        print("=" * 60)

    except Exception as error:
        print(
            f"{datetime.datetime.now().isoformat()} | Training | ISBN: {isbn}"
            f" | Error while downloading book: {error}\n{traceback.format_exc()}"
        )

print(f"{datetime.datetime.now().isoformat()} | Training | End of the script")
