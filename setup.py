#!/usr/bin/env python3
"""
Setup script for the Legal Contract Analyzer
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Legal Contract Analyzer")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create virtual environment
    if not Path("venv").exists():
        print("\n📦 Creating virtual environment...")
        if not run_command("python -m venv venv", "Creating virtual environment"):
            sys.exit(1)
    
    # Determine activation command
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        sys.exit(1)
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing requirements"):
        sys.exit(1)
    
    # Download spaCy model
    print("\n🧠 Downloading spaCy model...")
    if not run_command(f"{pip_cmd} install spacy", "Installing spaCy"):
        sys.exit(1)
    
    if not run_command(f"{pip_cmd.replace('pip', 'python')} -m spacy download en_core_web_sm", "Downloading spaCy English model"):
        sys.exit(1)
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    directories = ["data", "models", "logs", "data/legal_knowledge"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Test installation
    print("\n🧪 Testing installation...")
    try:
        # Test imports using virtual environment Python
        test_script = """
import sys
import torch
import transformers
import streamlit
import langchain
print("All required packages imported successfully")

# Test CUDA availability
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available - will use CPU")
"""
        # Write test script to temporary file
        test_file = "temp_test_imports.py"
        with open(test_file, "w") as f:
            f.write(test_script)
        
        # Run test using virtual environment Python
        python_cmd = pip_cmd.replace("pip", "python")
        if not run_command(f"{python_cmd} {test_file}", "Testing package imports"):
            # Clean up
            if os.path.exists(test_file):
                os.remove(test_file)
            sys.exit(1)
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
            
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Run the test script:")
    print("   python test_analyzer.py")
    print("3. Start the web application:")
    print("   streamlit run streamlit_app.py")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 