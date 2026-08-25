import pdfplumber
import pandas as pd
import re
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# -------- FILE PATHS --------
pdf_path = r"D:\icd\DOC-20260225-WA0017..pdf"
output_dir = r"D:\icd\output"
os.makedirs(output_dir, exist_ok=True)

output_csv = os.path.join(output_dir, "output.csv")

# -------- PRECOMPILED REGEX --------
reg_no_pattern = re.compile(r"Reg No\s*:\s*(\d+)")
reg_date_pattern = re.compile(r"Reg date\s*:\s*([0-9/]+)")
sex_pattern = re.compile(r"Sex\s*:\s*([MF])")

# -------- FUNCTION --------
def extract_records(text):
    records = []
    lines = text.split("\n")

    for i in range(1, len(lines)):
        line = lines[i].strip()

        if "Reg No" in line and "Reg date" in line and "Sex" in line:

            name = lines[i - 1].strip()

            if len(name) < 3 or "MAHARASHTRA" in name or name.startswith("Sr"):
                continue

            reg_no = reg_no_pattern.search(line)
            reg_date = reg_date_pattern.search(line)
            sex = sex_pattern.search(line)

            records.append({
                "Name": name,
                "Sex": sex.group(1) if sex else "",
                "Reg No": reg_no.group(1) if reg_no else "",
                "Reg Date": reg_date.group(1) if reg_date else ""
            })

    return records


# -------- PAGE PROCESSOR --------
def process_page(page_index):
    results = []

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]

        width, height = page.width, page.height

        left = page.crop((0, 0, width * 0.48, height))
        right = page.crop((width * 0.52, 0, width, height))

        left_text = left.extract_text() or ""
        right_text = right.extract_text() or ""

        left_records = extract_records(left_text)
        right_records = extract_records(right_text)

        max_len = max(len(left_records), len(right_records))

        for i in range(max_len):
            if i < len(left_records):
                results.append(left_records[i])

            if i < len(right_records):
                results.append(right_records[i])

    return results


# -------- MAIN --------
def main():
    total_pages = len(pdfplumber.open(pdf_path).pages)

    print(f"Total Pages: {total_pages}")
    print(f"Using {cpu_count()} CPU cores 🚀")

    chunk_size = 500  # write every 500 pages

    with open(output_csv, "w", newline='', encoding='utf-8') as f:
        header_written = False
        sr_no = 1

        for start in range(0, total_pages, chunk_size):
            end = min(start + chunk_size, total_pages)

            with Pool(cpu_count()) as pool:
                results = list(tqdm(
                    pool.imap(process_page, range(start, end)),
                    total=(end - start),
                    desc=f"Processing pages {start}-{end}"
                ))

            flat_data = []
            for page_records in results:
                for rec in page_records:
                    flat_data.append({
                        "Sr No": sr_no,
                        "Name of RMP": rec["Name"],
                        "Sex": rec["Sex"],
                        "Reg No": rec["Reg No"],
                        "Reg Date": rec["Reg Date"]
                    })
                    sr_no += 1

            df = pd.DataFrame(flat_data)

            df.to_csv(
                f,
                index=False,
                header=not header_written,
                mode='a'
            )

            header_written = True

    print("✅ Done! Data saved progressively.")


if __name__ == "__main__":
    main()