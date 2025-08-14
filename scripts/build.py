#!/usr/bin/env python3
"""
Build script for AgroGraphNet package
"""
import subprocess
import sys
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔨 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Command: {cmd}")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def clean_build():
    """Clean previous build artifacts"""
    print("🧹 Cleaning previous build artifacts...")
    
    dirs_to_clean = ['build', 'dist', 'agrographnet.egg-info']
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"   Removed {dir_name}/")
    
    print("✅ Cleanup completed")

def install_build_tools():
    """Install required build tools"""
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    run_command("python -m pip install --upgrade build twine", "Installing build tools")

def run_tests():
    """Run tests before building"""
    print("🧪 Running tests...")
    try:
        subprocess.run("python -m pytest tests/ -v", shell=True, check=True)
        print("✅ All tests passed")
    except subprocess.CalledProcessError:
        print("⚠️  Some tests failed, but continuing with build...")
    except FileNotFoundError:
        print("ℹ️  No tests found, skipping test phase")

def build_package():
    """Build the package"""
    run_command("python -m build", "Building package")

def check_package():
    """Check the built package"""
    run_command("python -m twine check dist/*", "Checking package")

def main():
    """Main build process"""
    print("🚀 Starting AgroGraphNet package build process")
    print("=" * 50)
    
    # Step 1: Clean previous builds
    clean_build()
    
    # Step 2: Install build tools
    install_build_tools()
    
    # Step 3: Run tests (optional)
    run_tests()
    
    # Step 4: Build package
    build_package()
    
    # Step 5: Check package
    check_package()
    
    print("\n" + "=" * 50)
    print("🎉 Build process completed successfully!")
    print("\nNext steps:")
    print("1. Test installation locally:")
    print("   pip install dist/agrographnet-*.whl")
    print("\n2. Upload to TestPyPI:")
    print("   python -m twine upload --repository testpypi dist/*")
    print("\n3. Upload to PyPI:")
    print("   python -m twine upload dist/*")

if __name__ == "__main__":
    main()