#!/usr/bin/env python3
"""
System verification script for M4 Pro setup.
Checks Python version, packages, and Metal acceleration.
"""

import sys
import platform

def check_python_version():
    """Check if Python 3.10 is being used."""
    print("=" * 60)
    print("1. Checking Python Version")
    print("=" * 60)
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Python version: {version_str}")
    
    if version.major == 3 and version.minor == 10:
        print("✅ Python 3.10 detected - Optimal!")
    elif version.major == 3 and version.minor >= 8:
        print(f"⚠️  Python {version.major}.{version.minor} - Compatible but 3.10 recommended")
    else:
        print(f"❌ Python {version.major}.{version.minor} - Upgrade to 3.10 recommended")
    
    print(f"Architecture: {platform.machine()}")
    if platform.machine() == "arm64":
        print("✅ ARM64 architecture detected (Apple Silicon)")
    else:
        print(f"⚠️  Non-ARM architecture: {platform.machine()}")
    print()


def check_packages():
    """Check if required packages are installed."""
    print("=" * 60)
    print("2. Checking Required Packages")
    print("=" * 60)
    
    required_packages = {
        "torch": "PyTorch (for neural networks)",
        "transformers": "HuggingFace Transformers",
        "sentence_transformers": "Sentence Transformers",
        "faiss": "FAISS (vector search)",
        "datasets": "HuggingFace Datasets",
        "nltk": "NLTK (tokenization)",
        "rank_bm25": "BM25 implementation",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "tqdm": "Progress bars",
    }
    
    all_installed = True
    
    for package, description in required_packages.items():
        try:
            if package == "faiss":
                # FAISS has different import name
                import faiss
                version = faiss.__version__ if hasattr(faiss, '__version__') else "installed"
            else:
                mod = __import__(package)
                version = mod.__version__ if hasattr(mod, '__version__') else "installed"
            
            print(f"✅ {package:20s} ({description}): {version}")
        except ImportError:
            print(f"❌ {package:20s} ({description}): NOT INSTALLED")
            all_installed = False
    
    print()
    if all_installed:
        print("✅ All required packages are installed!")
    else:
        print("❌ Some packages are missing. Run: pip install -r requirements.txt")
    print()


def check_torch_mps():
    """Check if PyTorch MPS (Metal) is available."""
    print("=" * 60)
    print("3. Checking Metal Performance Shaders (MPS)")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Check MPS availability
        if hasattr(torch.backends, 'mps'):
            is_available = torch.backends.mps.is_available()
            is_built = torch.backends.mps.is_built()
            
            print(f"MPS built: {is_built}")
            print(f"MPS available: {is_available}")
            
            if is_available:
                print("✅ Metal acceleration is READY!")
                print("   Your M4 Pro will use GPU for encoding")
                print("   Expected speedup: 3-4x faster than CPU")
            else:
                if not is_built:
                    print("❌ MPS not built in this PyTorch version")
                    print("   Solution: pip install --upgrade torch")
                else:
                    print("⚠️  MPS built but not available")
                    print("   This might be normal on non-Apple Silicon Macs")
        else:
            print("❌ MPS backend not found in PyTorch")
            print("   Your PyTorch version might be too old")
            print("   Solution: pip install --upgrade torch")
        
        # Check CUDA (shouldn't be available on Mac, but check anyway)
        if torch.cuda.is_available():
            print("ℹ️  CUDA is available (unexpected on Mac)")
        
    except ImportError:
        print("❌ PyTorch is not installed")
        print("   Solution: pip install torch")
    
    print()


def check_device_detection():
    """Check if device detection works."""
    print("=" * 60)
    print("4. Testing Device Detection")
    print("=" * 60)
    
    try:
        import torch
        
        # Simulate the get_device function
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        print(f"Detected device: {device}")
        
        if device == "mps":
            print("✅ Will use Apple Silicon GPU (Metal)")
        elif device == "cuda":
            print("✅ Will use NVIDIA GPU")
        else:
            print("⚠️  Will use CPU (slower)")
            print("   Check if MPS is available above")
        
    except Exception as e:
        print(f"❌ Error detecting device: {e}")
    
    print()


def check_nltk_data():
    """Check if NLTK data is downloaded."""
    print("=" * 60)
    print("5. Checking NLTK Data")
    print("=" * 60)
    
    try:
        import nltk
        
        # Check for punkt tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
            print("✅ NLTK punkt tokenizer: installed")
        except LookupError:
            print("❌ NLTK punkt tokenizer: NOT FOUND")
            print("   Solution: python -c \"import nltk; nltk.download('punkt')\"")
    
    except ImportError:
        print("❌ NLTK not installed")
    
    print()


def test_sentence_transformer():
    """Test if sentence transformer works with MPS."""
    print("=" * 60)
    print("6. Testing Sentence Transformer with MPS")
    print("=" * 60)
    
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        import time
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Testing with device: {device}")
        
        print("Loading model (this may take a moment)...")
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=device
        )
        
        print("Encoding test sentences...")
        test_texts = ["This is a test sentence"] * 100
        
        start = time.time()
        embeddings = model.encode(
            test_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        elapsed = time.time() - start
        
        print(f"✅ Successfully encoded {len(test_texts)} sentences")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Speed: {len(test_texts)/elapsed:.0f} sentences/sec")
        print(f"   Embedding shape: {embeddings.shape}")
        
        if device == "mps":
            print(f"   🚀 Using Metal acceleration!")
        
    except Exception as e:
        print(f"❌ Error testing sentence transformer: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def check_disk_space():
    """Check available disk space."""
    print("=" * 60)
    print("7. Checking Disk Space")
    print("=" * 60)
    
    try:
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        
        print(f"Total: {total // (2**30)} GB")
        print(f"Used: {used // (2**30)} GB")
        print(f"Free: {free // (2**30)} GB")
        
        if free > 50 * (2**30):  # 50 GB
            print("✅ Plenty of disk space available")
        elif free > 20 * (2**30):  # 20 GB
            print("⚠️  Disk space is adequate but monitor usage")
        else:
            print("❌ Low disk space - consider freeing up space")
        
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
    
    print()


def check_memory():
    """Check system memory."""
    print("=" * 60)
    print("8. Checking System Memory")
    print("=" * 60)
    
    try:
        import psutil
        
        mem = psutil.virtual_memory()
        
        print(f"Total RAM: {mem.total / (2**30):.1f} GB")
        print(f"Available: {mem.available / (2**30):.1f} GB")
        print(f"Used: {mem.used / (2**30):.1f} GB ({mem.percent}%)")
        
        if mem.total >= 24 * (2**30):
            print("✅ 24GB RAM detected - Perfect for this project!")
        elif mem.total >= 16 * (2**30):
            print("✅ 16GB+ RAM - Good for this project")
        elif mem.total >= 8 * (2**30):
            print("⚠️  8GB RAM - Adequate but may need to reduce batch sizes")
        else:
            print("⚠️  Limited RAM - Use smaller batch sizes")
        
    except ImportError:
        print("ℹ️  psutil not installed, skipping memory check")
        print("   (Optional: pip install psutil)")
    except Exception as e:
        print(f"⚠️  Could not check memory: {e}")
    
    print()


def print_summary():
    """Print summary and next steps."""
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
If all checks passed (✅), you're ready to start!

Next steps:
1. Download datasets:
   bash scripts/download_datasets.sh data

2. Preprocess data:
   python scripts/preprocess_data.py --wiki_max 10000

3. Build indexes:
   python scripts/chunk_and_index.py --batch_size 128 --device mps

4. Test retrieval:
   python scripts/smoke_test.py --device mps

See M4_QUICKSTART.md for complete commands.
    """)


def main():
    print("\n")
    print("*" * 60)
    print("M4 Pro System Verification")
    print("Hybrid RAG Project Setup Check")
    print("*" * 60)
    print("\n")
    
    check_python_version()
    check_packages()
    check_torch_mps()
    check_device_detection()
    check_nltk_data()
    test_sentence_transformer()
    check_disk_space()
    check_memory()
    print_summary()


if __name__ == "__main__":
    main()
