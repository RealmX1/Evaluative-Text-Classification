
import csv
import re
import os

from sklearn.datasets import fetch_20newsgroups
import pandas as pd

def twenty_newsgroup_to_csv(output_path):
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df.columns = ['text', 'target']

    targets = pd.DataFrame( newsgroups_train.target_names)
    targets.columns=['title']

    out = pd.merge(df, targets, left_on='target', right_index=True)
    out['date'] = pd.to_datetime('now')
    out.to_csv(output_path, index=False)


def split_into_paragraphs(text):
    """Split text into paragraphs based on two consecutive newlines."""
    if not text:
        return []
    # Split by two or more newlines
    paragraphs = re.split(r'\n\s*\n', text)
    # Filter out empty paragraphs
    return [p.strip() for p in paragraphs if p.strip()]

def split_paragraph_by_sentences(paragraph, max_length=1000):
    """Split a paragraph into chunks of maximum length, breaking at sentence boundaries."""
    if len(paragraph) <= max_length:
        return [paragraph]
    
    # Pattern to match sentence endings (period, question mark, exclamation mark followed by space or end of string)
    sentence_pattern = r'[.!?](?:\s|$)'
    
    chunks = []
    start_idx = 0
    current_length = 0
    last_sentence_end = 0
    
    # Find all sentence endings
    for match in re.finditer(sentence_pattern, paragraph):
        sentence_end = match.end()
        sentence_length = sentence_end - start_idx
        
        if current_length + sentence_length <= max_length:
            # If adding this sentence doesn't exceed max_length, continue accumulating
            current_length += sentence_length
            last_sentence_end = sentence_end
        else:
            # If adding this sentence would exceed max_length, create a chunk with accumulated sentences
            if last_sentence_end > start_idx:  # Make sure we've accumulated at least one sentence
                chunks.append(paragraph[start_idx:last_sentence_end].strip())
                start_idx = last_sentence_end
                current_length = sentence_end - start_idx
                last_sentence_end = sentence_end
            else:
                # If a single sentence is longer than max_length, we need to break it arbitrarily
                # First try to find a space near max_length
                space_before_max = paragraph.rfind(' ', start_idx, start_idx + max_length)
                if space_before_max != -1 and space_before_max > start_idx:
                    chunks.append(paragraph[start_idx:space_before_max].strip())
                    start_idx = space_before_max + 1
                else:
                    # If no space found, break at exactly max_length
                    chunks.append(paragraph[start_idx:start_idx + max_length].strip())
                    start_idx = start_idx + max_length
                
                # Reset tracking variables
                current_length = 0
                last_sentence_end = start_idx
    
    # Add any remaining text as the final chunk
    if start_idx < len(paragraph):
        chunks.append(paragraph[start_idx:].strip())
    
    return chunks

def split_text(input_path, output_path):
    """Process a CSV file, splitting the 'text' column according to the rules."""
    with open(input_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        if 'text' not in fieldnames:
            raise ValueError("CSV file must contain a 'text' column")
        
        rows = []
        for row in reader:
            original_text = row['text']
            paragraphs = split_into_paragraphs(original_text)
            
            for paragraph in paragraphs:
                if len(paragraph) > 1000:
                    chunks = split_paragraph_by_sentences(paragraph)
                    for chunk in chunks:
                        new_row = row.copy()
                        new_row['text'] = chunk
                        rows.append(new_row)
                else:
                    new_row = row.copy()
                    new_row['text'] = paragraph
                    rows.append(new_row)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Processed CSV saved to {output_path}")
    print(f"Original rows: {reader.line_num - 1}, New rows: {len(rows)}")

def main():
    
    input_path = os.path.join('data', '20_newsgroup.csv')
    output_path = os.path.join('data', '20_newsgroup_pruned.csv')
    
    twenty_newsgroup_to_csv(input_path)
    
    # Check if the input file exists, if not, create it
    if not os.path.exists(input_path):
        print(f"Crudely pruned '{input_path}' does not exist. Creating it...")
        twenty_newsgroup_to_csv()
    
    if not os.path.exists(output_path):
        print(f"Output file '{output_path}' does not exist. Creating it...")
        split_text(input_path, output_path)

if __name__ == "__main__":
    main()