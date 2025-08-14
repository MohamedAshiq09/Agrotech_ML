#!/usr/bin/env python3
"""
Release automation script for AgroGraphNet
"""
import subprocess
import sys
import json
from pathlib import Path

def run_command(cmd, description):
    """Run shell command and handle errors"""
    print(f"🔨 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def get_version():
    """Get current version from package"""
    version_file = Path("agrographnet/__version__.py")
    with open(version_file) as f:
        content = f.read()
        for line in content.split('\n'):
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    raise ValueError("Version not found")

def check_git_status():
    """Check if git working directory is clean"""
    try:
        result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            print("❌ Git working directory is not clean. Please commit all changes first.")
            print("Uncommitted changes:")
            print(result.stdout)
            sys.exit(1)
        print("✅ Git working directory is clean")
    except subprocess.CalledProcessError:
        print("⚠️  Could not check git status")

def create_git_tag(version):
    """Create and push git tag"""
    tag_name = f"v{version}"
    run_command(f"git tag {tag_name}", f"Creating git tag {tag_name}")
    run_command(f"git push origin {tag_name}", f"Pushing tag {tag_name}")

def test_package_locally():
    """Test package installation locally"""
    print("🧪 Testing package installation locally...")
    
    # Create a temporary virtual environment for testing
    run_command("python -m venv test_env", "Creating test environment")
    
    # Install the package
    if sys.platform == "win32":
        pip_cmd = "test_env\\Scripts\\pip"
    else:
        pip_cmd = "test_env/bin/pip"
    
    run_command(f"{pip_cmd} install dist/agrographnet-*.whl", "Installing package in test environment")
    
    # Test CLI command
    if sys.platform == "win32":
        agrographnet_cmd = "test_env\\Scripts\\agrographnet"
    else:
        agrographnet_cmd = "test_env/bin/agrographnet"
    
    run_command(f"{agrographnet_cmd} --help", "Testing CLI command")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_env")
    print("✅ Local package test completed")

def upload_to_testpypi():
    """Upload package to TestPyPI"""
    print("📦 Uploading to TestPyPI...")
    run_command("python -m twine upload --repository testpypi dist/*", "Uploading to TestPyPI")
    
    version = get_version()
    print(f"✅ Package uploaded to TestPyPI!")
    print(f"Test installation with: pip install -i https://test.pypi.org/simple/ agrographnet=={version}")

def upload_to_pypi():
    """Upload package to PyPI"""
    response = input("🚨 Are you sure you want to upload to PyPI? This cannot be undone! (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Upload cancelled")
        return
    
    print("📦 Uploading to PyPI...")
    run_command("python -m twine upload dist/*", "Uploading to PyPI")
    
    version = get_version()
    print(f"🎉 Package uploaded to PyPI!")
    print(f"Install with: pip install agrographnet=={version}")

def main():
    """Main release process"""
    print("🚀 Starting AgroGraphNet release process")
    print("=" * 50)
    
    # Get current version
    version = get_version()
    print(f"📋 Current version: {version}")
    
    # Check git status
    check_git_status()
    
    # Build package
    print("\n1️⃣ Building package...")
    run_command("python scripts/build.py", "Running build script")
    
    # Test locally
    print("\n2️⃣ Testing package locally...")
    test_package_locally()
    
    # Create git tag
    print("\n3️⃣ Creating git tag...")
    create_git_tag(version)
    
    # Upload to TestPyPI
    print("\n4️⃣ Uploading to TestPyPI...")
    upload_to_testpypi()
    
    # Ask about PyPI upload
    print("\n5️⃣ PyPI Upload")
    upload_choice = input("Upload to PyPI? (y/n): ").lower()
    if upload_choice == 'y':
        upload_to_pypi()
    else:
        print("⏭️  Skipping PyPI upload")
        print(f"To upload later, run: python -m twine upload dist/*")
    
    print("\n" + "=" * 50)
    print("🎉 Release process completed!")
    print(f"\n📦 AgroGraphNet v{version} is now available!")
    print("\n📋 Post-release checklist:")
    print("1. Update documentation if needed")
    print("2. Announce release on social media/blog")
    print("3. Update any dependent projects")
    print("4. Monitor for issues and user feedback")

if __name__ == "__main__":
    main()