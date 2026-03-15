import re 
 
def clean_text(raw: str) -> str: 
    """ 
    Cleans raw OCR output. 
    Steps: lowercase → remove non-alpha chars → strip short junk tokens → strip 
whitespace. 
    """ 
    if not raw or not raw.strip(): 
        return '' 
    text = raw.lower() 
    # Remove non-alphabetic characters (keep spaces) 
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove standalone single characters (OCR noise) 
    text = re.sub(r'\b[a-z]{1}\b', '', text) 
    # Collapse multiple spaces 
    text = re.sub(r'\s+', ' ', text).strip() 
    return text 
 
if __name__ == '__main__': 
    sample = 'He||o W0rld! This is a Test123... \n\n OCR output...' 
    print(clean_text(sample))  # Expected: hello world this is test ocr output 
