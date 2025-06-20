#!/usr/bin/env python3
"""
Quick Start Script for M3 Mac
Automatically optimizes configuration and runs the evolutionary pipeline.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Evolutionary Text Generation - M3 Optimized        ║
║                      Quick Start Script                      ║
╚══════════════════════════════════════════════════════════════╝
""")

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  WARNING: Virtual environment not detected. Please activate your venv first.")
        print("   Run: source venv/bin/activate")
        return False
    
    # Check for required files
    required_files = [
        "src/main.py",
        "config/modelConfig.yaml", 
        "data/prompt.xlsx",
        ".env"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        if ".env" in missing_files:
            print("   Create .env file with your OpenAI API key:")
            print("   OPENAI_API_KEY=your_key_here")
        if "data/prompt.xlsx" in missing_files:
            print("   Add your seed prompts to data/prompt.xlsx")
        return False
    
    print("✅ All requirements met!")
    return True

def optimize_for_m3():
    """Run M3 optimization"""
    print("⚡ Optimizing configuration for M3 Mac...")
    
    try:
        result = subprocess.run([
            sys.executable, "src/utils/m3_optimizer.py", "--optimize-config"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Configuration optimized successfully!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Optimization failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        return False

def run_pipeline(generations=5):
    """Run the main pipeline"""
    print(f"🚀 Starting evolutionary pipeline (max {generations} generations)...")
    print("📊 Monitor progress in the logs and outputs/ directory")
    print("⏱️  This may take several minutes to hours depending on your settings")
    print()
    
    try:
        # Run the main pipeline
        cmd = [sys.executable, "src/main.py", "--generations", str(generations)]
        
        print(f"Running: {' '.join(cmd)}")
        print("=" * 60)
        
        # Run with real-time output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        start_time = time.time()
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        elapsed_time = time.time() - start_time
        
        if process.returncode == 0:
            print(f"\n✅ Pipeline completed successfully in {elapsed_time:.1f} seconds!")
            print_results_summary()
            return True
        else:
            print(f"\n❌ Pipeline failed with return code {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        return False

def print_results_summary():
    """Print a summary of results"""
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    
    # Check for results files
    results_files = [
        ("outputs/Population.json", "Final population"),
        ("outputs/EvolutionStatus.json", "Evolution status"),
        ("outputs/final_statistics.json", "Final statistics"),
        ("outputs/successful_genomes_gen_*.json", "Successful genomes")
    ]
    
    for file_pattern, description in results_files:
        if "*" in file_pattern:
            # Handle glob patterns
            import glob
            matches = glob.glob(file_pattern)
            if matches:
                print(f"✅ {description}: {len(matches)} files found")
            else:
                print(f"📄 {description}: No files found")
        else:
            if Path(file_pattern).exists():
                file_size = Path(file_pattern).stat().st_size / 1024  # KB
                print(f"✅ {description}: {file_pattern} ({file_size:.1f} KB)")
            else:
                print(f"📄 {description}: Not found")
    
    print("\n🔍 Next steps:")
    print("1. Check outputs/ directory for detailed results")
    print("2. Open experiments/experiments.ipynb for analysis")
    print("3. Run python src/utils/m3_optimizer.py --report for performance report")

def main():
    """Main execution flow"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please fix the issues above and try again.")
        sys.exit(1)
    
    # Optimize for M3
    if not optimize_for_m3():
        print("\n❌ Configuration optimization failed. Continuing with default settings...")
        input("Press Enter to continue or Ctrl+C to exit...")
    
    # Ask user for generations
    try:
        generations_input = input("\n🎯 How many generations to run? (default: 5, 0 for unlimited): ")
        if generations_input.strip() == "":
            generations = 5
        elif generations_input.strip() == "0":
            generations = None
        else:
            generations = int(generations_input)
    except ValueError:
        print("Invalid input, using default: 5 generations")
        generations = 5
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    
    # Run the pipeline
    success = run_pipeline(generations)
    
    if success:
        print("\n🎉 Evolution completed! Check the results above.")
    else:
        print("\n💔 Something went wrong. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 