# Combined dataset loader and tokenizer for multiple conversational datasets
# Install dependencies: pip install datasets gitpython transformers

from datasets import load_dataset
from transformers import GPT2Tokenizer
import os
import git

def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {
        'pad_token': '[PAD]',
        'bos_token': '[BOS]',
        'eos_token': '[EOS]',
        'additional_special_tokens': ['[CUSTOM1]', '[CUSTOM2]']
    }
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def tokenize_and_save(dataset, tokenizer, text_fields, output_file):
    max_length = 1024
    count = 0
    def extract_texts(obj):
        # Recursively extract all string fields from dicts/lists
        if isinstance(obj, dict):
            for v in obj.values():
                yield from extract_texts(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from extract_texts(v)
        elif isinstance(obj, str):
            yield obj
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            found = False
            # Try user-specified fields first
            for field in text_fields:
                if field in item and item[field]:
                    for text in extract_texts(item[field]):
                        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
                        f.write(' '.join(map(str, tokens)) + '\n')
                        count += 1
                        found = True
            # If nothing found, try all text fields recursively
            if not found:
                for text in extract_texts(item):
                    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
                    f.write(' '.join(map(str, tokens)) + '\n')
                    count += 1
    print(f"Tokenized {count} lines to {output_file}")

def process_hf_dataset(dataset_name, text_fields, split='train', output_file='output.txt'):
    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split=split)
    tokenizer = get_tokenizer()
    tokenize_and_save(ds, tokenizer, text_fields, output_file)
    print(f"Tokenized data saved to {output_file}")

def clone_and_process_git_repo(repo_url, process_func):
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    if not os.path.exists(repo_name):
        print(f"Cloning {repo_url}...")
        git.Repo.clone_from(repo_url, repo_name)
    process_func(repo_name)

if __name__ == "__main__":
    # Hugging Face datasets (using official names where possible)
    hf_datasets = [
        ("roskoN/dailydialog", ["dialog"]),
        ("Cynaptics/persona-chat", ["utterance", "persona"]),
        ("tatsu-lab/alpaca", ["text", "instruction", "input", "output"]),
        ("OpenAssistant/oasst1", ["text", "message"]),
        ("ParlAI/blended_skill_talk", ["text", "context"]),
        ("kaistlayner/empathy-dataset", ["text", "context"]),
    ]
    for ds_name, fields in hf_datasets:
        try:
            process_hf_dataset(ds_name, fields, output_file=f"tokenized_{ds_name.replace('/', '_')}.txt")
        except Exception as e:
            print(f"Failed to process {ds_name}: {e}")
    
    # Knowledge datasets (Wikipedia + Wizard of Wikipedia)
    print("\n=== Loading Knowledge Datasets ===")
    
    # 1. Wikipedia (FILTERED - only important/useful articles)
    try:
        print("Loading Wikipedia (filtered for useful content)...")
        wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        tokenizer = get_tokenizer()
        
        # Important categories for Jarvis AI assistant
        important_keywords = [
            "technology", "computer", "software", "hardware", "internet", "programming",
            "science", "physics", "chemistry", "biology", "mathematics",
            "history", "world war", "ancient", "empire", "civilization",
            "geography", "country", "city", "continent", "ocean",
            "medicine", "health", "disease", "treatment", "anatomy",
            "art", "music", "literature", "film", "painting",
            "philosophy", "psychology", "economics", "politics", "law",
            "sports", "game", "olympic", "football", "cricket",
            "food", "cooking", "recipe", "nutrition",
            "travel", "tourism", "culture", "language", "religion",
            "business", "finance", "market", "company", "industry",
            "education", "university", "school", "learning", "research",
            "nature", "animal", "plant", "weather", "climate", "environment"
        ]
        
        with open("tokenized_wikipedia.txt", 'w', encoding='utf-8') as f:
            count = 0
            max_articles = 500000  # Limit to 500K most relevant articles
            
            for item in wiki:
                if count >= max_articles:
                    break
                    
                title = item.get("title", "").lower()
                text = item.get("text", "")
                
                # Only keep articles that match important topics
                is_important = any(keyword in title or keyword in text[:500].lower() 
                                  for keyword in important_keywords)
                
                # Skip very short or low quality articles
                if not is_important or len(text) < 200:
                    continue
                
                if text:
                    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=1024)
                    f.write(' '.join(map(str, tokens)) + '\n')
                    count += 1
                    
                    if count % 10000 == 0:
                        print(f"  Tokenized {count} relevant Wikipedia articles...")
            
            print(f"Tokenized {count} FILTERED Wikipedia articles to tokenized_wikipedia.txt")
    except Exception as e:
        print(f"Failed to process Wikipedia: {e}")
    
    # 2. Wizard of Wikipedia (knowledge-grounded conversation - ALL)
    try:
        print("Loading Wizard of Wikipedia...")
        wow = load_dataset("wow", split="train", streaming=True)
        tokenizer = get_tokenizer()
        
        with open("tokenized_wow.txt", 'w', encoding='utf-8') as f:
            count = 0
            max_entries = 100000  # Limit to 100K entries
            
            for item in wow:
                if count >= max_entries:
                    break
                    
                for utterance in item.get("utterances", []):
                    if isinstance(utterance, str) and utterance and len(utterance) > 20:
                        tokens = tokenizer.encode(utterance, add_special_tokens=True, truncation=True, max_length=1024)
                        f.write(' '.join(map(str, tokens)) + '\n')
                        count += 1
                
                if count % 10000 == 0:
                    print(f"  Tokenized {count} Wizard of Wikipedia utterances...")
            
            print(f"Tokenized {count} Wizard of Wikipedia utterances to tokenized_wow.txt")
    except Exception as e:
        print(f"Failed to process Wizard of Wikipedia: {e}")

    # GitHub datasets (custom processing required)
    def process_opensubtitles(repo_dir):
        # Example: parse .txt files in the repo for subtitles
        tokenizer = get_tokenizer()
        for root, _, files in os.walk(repo_dir):
            for file in files:
                if file.endswith('.txt'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f_in, \
                         open(f"tokenized_opensubtitles.txt", 'a', encoding='utf-8') as f_out:
                        for line in f_in:
                            tokens = tokenizer.encode(line.strip(), add_special_tokens=True)
                            f_out.write(' '.join(map(str, tokens)) + '\n')
    clone_and_process_git_repo('https://github.com/domerin0/opensubtitles-parser.git', process_opensubtitles)

    def process_multiwoz(repo_dir):
        # Example: parse .json files for dialogue
        import json
        tokenizer = get_tokenizer()
        for root, _, files in os.walk(repo_dir):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f_in, \
                         open(f"tokenized_multiwoz.txt", 'a', encoding='utf-8') as f_out:
                        try:
                            data = json.load(f_in)
                            for k, v in data.items():
                                if isinstance(v, str):
                                    tokens = tokenizer.encode(v, add_special_tokens=True)
                                    f_out.write(' '.join(map(str, tokens)) + '\n')
                        except Exception:
                            continue
    clone_and_process_git_repo('https://github.com/budzianowski/multiwoz.git', process_multiwoz)
