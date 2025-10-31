"""
Preprocess and prepare datasets for Hybrid RAG pipeline.
Downloads and prepares HotpotQA, Natural Questions, and Wikipedia passages.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd


def download_and_prepare_hotpotqa(data_dir: Path):
    """Process HotpotQA dataset into a unified format."""
    print("\n" + "="*50)
    print("Processing HotpotQA dataset")
    print("="*50)
    
    hotpot_dir = data_dir / "hotpotqa"
    processed_dir = data_dir / "processed" / "hotpotqa"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train and dev splits
    for split in ["train", "dev"]:
        if split == "train":
            input_file = hotpot_dir / "hotpot_train_v1.1.json"
        else:
            input_file = hotpot_dir / "hotpot_dev_distractor_v1.json"
        
        if not input_file.exists():
            print(f"  Warning: {input_file} not found. Run download script first.")
            continue
        
        print(f"\n  Processing {split} split...")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        processed_examples = []
        
        for example in tqdm(data, desc=f"  Processing {split}"):
            processed = {
                "id": example["_id"],
                "question": example["question"],
                "answer": example["answer"],
                "type": example["type"],  # bridge, comparison
                "level": example["level"],  # easy, medium, hard
                "supporting_facts": example["supporting_facts"],
                "context": example["context"]  # List of [title, sentences]
            }
            processed_examples.append(processed)
        
        # Save processed data
        output_file = processed_dir / f"{split}.json"
        with open(output_file, 'w') as f:
            json.dump(processed_examples, f, indent=2)
        
        print(f"  Saved {len(processed_examples)} examples to {output_file}")


def download_and_prepare_nq(data_dir: Path, sample_size: int = None):
    """Download and prepare Natural Questions dataset."""
    print("\n" + "="*50)
    print("Processing Natural Questions dataset")
    print("="*50)
    
    processed_dir = data_dir / "processed" / "nq"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Download NQ-Open (simplified version for open-domain QA)
    print("\n  Downloading NQ-Open from HuggingFace...")
    
    for split in ["train", "validation"]:
        print(f"\n  Processing {split} split...")
        
        # Load from HuggingFace
        dataset = load_dataset("nq_open", split=split)
        
        if sample_size and split == "train":
            dataset = dataset.select(range(min(sample_size, len(dataset))))
        
        processed_examples = []
        
        for example in tqdm(dataset, desc=f"  Processing {split}"):
            processed = {
                "id": f"nq_{split}_{len(processed_examples)}",
                "question": example["question"],
                "answer": example["answer"]  # List of acceptable answers
            }
            processed_examples.append(processed)
        
        # Save processed data
        output_split = "dev" if split == "validation" else split
        output_file = processed_dir / f"{output_split}.json"
        with open(output_file, 'w') as f:
            json.dump(processed_examples, f, indent=2)
        
        print(f"  Saved {len(processed_examples)} examples to {output_file}")


def download_and_prepare_wikipedia(data_dir: Path, max_passages: int = 100000):
    """Download and prepare Wikipedia passages for retrieval corpus."""
    print("\n" + "="*50)
    print("Processing Wikipedia corpus")
    print("="*50)
    
    processed_dir = data_dir / "processed" / "wikipedia"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n  Downloading Wikipedia from HuggingFace...")
    print(f"  (Limited to {max_passages} passages for initial experiments)")
    
    # Use the Wikipedia dataset (simplified version)
    # For full experiments, you might want to use the full dump
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    
    passages = []
    passage_id = 0
    
    print("\n  Processing Wikipedia articles...")
    for i, article in enumerate(tqdm(dataset, total=max_passages, desc="  Processing")):
        if i >= max_passages:
            break
        
        # Split article into passages (simple paragraph-based splitting)
        title = article["title"]
        text = article["text"]
        
        # Split by paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        for para in paragraphs:
            if len(para.split()) < 20:  # Skip very short paragraphs
                continue
            
            passages.append({
                "id": f"wiki_{passage_id}",
                "title": title,
                "text": para,
                "source": "wikipedia"
            })
            passage_id += 1
    
    # Save passages
    output_file = processed_dir / "passages.jsonl"
    with open(output_file, 'w') as f:
        for passage in passages:
            f.write(json.dumps(passage) + "\n")
    
    print(f"\n  Saved {len(passages)} passages to {output_file}")
    
    # Create a metadata file
    metadata = {
        "num_passages": len(passages),
        "source": "wikipedia_20220301",
        "max_articles_processed": max_passages
    }
    
    with open(processed_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def create_corpus_from_hotpotqa(data_dir: Path):
    """Extract passages from HotpotQA contexts to add to corpus."""
    print("\n" + "="*50)
    print("Extracting passages from HotpotQA contexts")
    print("="*50)
    
    hotpot_dir = data_dir / "processed" / "hotpotqa"
    corpus_dir = data_dir / "processed" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    passages = []
    passage_id = 0
    seen_passages = set()
    
    for split in ["train", "dev"]:
        split_file = hotpot_dir / f"{split}.json"
        if not split_file.exists():
            continue
        
        print(f"\n  Processing {split} split...")
        
        with open(split_file, 'r') as f:
            data = json.load(f)
        
        for example in tqdm(data, desc=f"  Extracting from {split}"):
            for title, sentences in example["context"]:
                # Combine sentences into passage
                passage_text = " ".join(sentences)
                
                # Deduplicate
                passage_key = f"{title}:::{passage_text}"
                if passage_key in seen_passages:
                    continue
                seen_passages.add(passage_key)
                
                passages.append({
                    "id": f"hotpot_{passage_id}",
                    "title": title,
                    "text": passage_text,
                    "source": "hotpotqa"
                })
                passage_id += 1
    
    # Save passages
    output_file = corpus_dir / "hotpotqa_passages.jsonl"
    with open(output_file, 'w') as f:
        for passage in passages:
            f.write(json.dumps(passage) + "\n")
    
    print(f"\n  Saved {len(passages)} unique passages to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for Hybrid RAG")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root data directory"
    )
    parser.add_argument(
        "--nq_sample",
        type=int,
        default=None,
        help="Sample size for NQ training data (None = use all)"
    )
    parser.add_argument(
        "--wiki_max",
        type=int,
        default=100000,
        help="Maximum number of Wikipedia articles to process"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa", "nq", "wikipedia"],
        choices=["hotpotqa", "nq", "wikipedia"],
        help="Which datasets to process"
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    print("="*50)
    print("Hybrid RAG: Dataset Preprocessing")
    print("="*50)
    print(f"Data directory: {data_dir.absolute()}")
    print(f"Datasets to process: {', '.join(args.datasets)}")
    print("="*50)
    
    # Process datasets
    if "hotpotqa" in args.datasets:
        download_and_prepare_hotpotqa(data_dir)
        create_corpus_from_hotpotqa(data_dir)
    
    if "nq" in args.datasets:
        download_and_prepare_nq(data_dir, sample_size=args.nq_sample)
    
    if "wikipedia" in args.datasets:
        download_and_prepare_wikipedia(data_dir, max_passages=args.wiki_max)
    
    print("\n" + "="*50)
    print("Preprocessing complete!")
    print("="*50)
    print("\nNext step: Run indexing script")
    print("  python scripts/chunk_and_index.py")


if __name__ == "__main__":
    main()
