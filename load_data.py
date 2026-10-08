from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd


DATASET_PATH = Path(__file__).with_name("dataset-uci.xlsx")


def column_index(cell_ref):
    """Convert an Excel cell reference such as 'AB12' to a zero-based column index."""
    letters = re.match(r"[A-Z]+", cell_ref).group()
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - 64
    return index - 1


def tag_name(tag):
    """Remove the namespace from an XML tag."""
    return tag.split("}")[-1]


def load_xlsx():
    try:
        df = pd.read_excel(DATASET_PATH)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    with zipfile.ZipFile(DATASET_PATH) as archive:
        names = archive.namelist()

        strings = []
        if "xl/sharedStrings.xml" in names:
            shared_strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_strings:
                text = "".join(
                    node.text or ""
                    for node in item.iter()
                    if tag_name(node.tag) == "t"
                )
                strings.append(text)

        sheets = sorted(
            name
            for name in names
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not sheets:
            raise ValueError(f"No worksheet found in Excel file: {DATASET_PATH}")

        rows = []
        worksheet = ET.fromstring(archive.read(sheets[0]))
        for row in worksheet.iter():
            if tag_name(row.tag) != "row":
                continue

            values = {}
            for cell in row:
                value = next(
                    (node.text for node in cell if tag_name(node.tag) == "v"),
                    None,
                )
                if value is None:
                    continue

                column = column_index(cell.get("r"))
                if cell.get("t") == "s":
                    values[column] = strings[int(value)]
                elif cell.get("t") == "str":
                    values[column] = value
                else:
                    values[column] = float(value)
            rows.append(values)

    nonempty_rows = [row for row in rows if row]
    if not nonempty_rows:
        raise ValueError(f"No table data found in Excel file: {DATASET_PATH}")

    n_columns = max(max(row) for row in nonempty_rows) + 1
    table = [[row.get(i) for i in range(n_columns)] for row in rows]
    df = pd.DataFrame(table[1:], columns=table[0])
    return df.apply(pd.to_numeric)


if __name__ == "__main__":
    df = load_xlsx()
    print(df.shape)
    print(df.iloc[:, 0].value_counts())