import arxiv
import re
import os
from transformers import pipeline
import pandas as pd
from pypdf import PdfReader
import shutil
import ast


list_papers = [
    "2410.08683",
    "2212.06181",
    "2502.00154",
    "2412.18578",
    "2312.15836",
    "2408.07677",
    "2302.13853",
    "2407.09942",
    "2502.00179",
    "1308.2928",
    "2509.05295",
    "2507.11536",
    "2508.05720",
    "2506.20655",
    "2506.19232",
    "1006.4395",
    "1812.06848",
    "1603.03148",
]

labels = [
    "quantum computing",
    "quantum information",
    "photonics",
    "randomized benchmarking",
    "classical shadows",
    "quantum error correction",
    "quantum chemistry",
    "tensor networks",
    "machine learning",
    "error mitigation",
    "noise",
    "quantum advantage"
]



def safe_name(s):
    """
        Clean up the names and shorten
    """
    assert type(s) == str, "safe_name: Input type shall be a string"
    s = re.sub(r'[\\/:*?"<>|]+', '_', s)   # Windows-safe
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:200]  # avoid overly long names


def download_from_list(list_papers, folder_name="raw_pdfs"):
    """
        Download from list of arxiv id numbers
    """
    os.makedirs(folder_name, exist_ok=True)
    for arx_number in list_papers:
        paper = next(arxiv.Client().results(arxiv.Search(arx_number)))
        paper_name = arx_number + ".pdf"
        paper.download_pdf(dirpath = folder_name, filename = paper_name)
        

def download_last_papers(num_papers = 10, folder_name = "raw_pdfs", category = "cat:quant-ph"):
    """
        Download last num_papers from a given arxiv category
    """
    # Create folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok = True)
    # Construct the default API client.
    client = arxiv.Client()
    # Search
    search = arxiv.Search(
        query = category,
        max_results = num_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    for res in client.results(search):
        arx_id = res.get_short_id()          # e.g. '2509.05295'
        # title  = res.title
        filename  = safe_name(f"{arx_id}.pdf")        
        res.download_pdf(dirpath = folder_name, filename = filename)

def extract_text_from_pdfs(folder_name):
    """
        Extract the text from the first page of a pdf
    """
    pdf_data = []
    pdf_files = [f for f in os.listdir(folder_name) if f.lower().endswith('pdf')]
    for pdf in pdf_files:
        reader = PdfReader(folder_name + "/" + pdf)
        pdf_name = pdf.strip(".pdf")
        first_page = reader.pages[0].extract_text() + "\n"
        pdf_info = {"arx_number": pdf_name, 
                    "text": first_page
                    }
        pdf_data.append(pdf_info)
    return pdf_data
        
        
def clean(text, max_chars=3000):
    """
        Preprocessing to clear new lines
    """
    if not text: 
        return ""
    text = str(text)
    text = " ".join(text.split()) # collapse whitespace/newlines
    return text[:max_chars] # keep the text short

def tag_text(classifier, text, labels=labels, threshold=0.3, highest_k_score=3):
    """
        Add tags from list of labels using BERT classifier
    """
    text = clean(text)
    # scores how well each tag fits, no training required
    out = classifier(text, labels, multi_label=True)
    pairs = list(zip(out["labels"], out["scores"]))  # already sorted high→low
    keep = [(l, float(s)) for l, s in pairs if s >= threshold]
    if not keep:
        keep = pairs[:highest_k_score]
    return keep
        
def classify_papers(papers, labels=labels, threshold=0.3, highest_k_score=3, 
                    csv_name="paper_tags.csv"):
    rows = []
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    for p in papers:
        tags = tag_text(classifier, p["text"], labels, threshold, highest_k_score)
        rows.append(
            {
            "arx_number": p["arx_number"],
            "tags": [t for t, _ in tags],
            "scores": [round(s, 3) for _, s in tags]
            }
            )
    df = pd.DataFrame(rows)
    df.to_csv("paper_tags.csv", index=False)
     

def move_files(source_folder, destination_folder, filenames):

    os.makedirs(destination_folder, exist_ok=True)
    for filename in filenames:
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.move(source_path, destination_path)
    
def filter_from_csv(filter_tag, csv_name="paper_tags.csv", source_folder="raw_pdfs", threshold=0.5):
   
    df = pd.read_csv(csv_name)
    df["scores"] = df["scores"].apply(ast.literal_eval)
    df["tags"] = df["tags"].apply(ast.literal_eval)
    filenames = []
    for _, row in df.iterrows():        
        if filter_tag in row['tags']:
            index = row['tags'].index(filter_tag)
            if row['scores'][index] >= threshold:
                filename = f"{row['arx_number']}.pdf"
                print(filename)
                filenames.append(filename)
                
    destination_folder = filter_tag
    move_files(source_folder, destination_folder, filenames)


















      
        
        
        
        
        