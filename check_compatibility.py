import sys
import platform
import pkg_resources
import subprocess
import os

def check_python_version():
    print("\n=== Checking Python Version ===")
    current_version = platform.python_version()
    required_version = "3.10.9"
    
    print(f"Current Python version: {current_version}")
    print(f"Required Python version: {required_version}")
    
    if current_version == required_version:
        print("✅ Python version matches exactly!")
        return True
    elif current_version.startswith("3.10."):
        print("⚠️ Python version is 3.10.x but not exactly 3.10.9. This might work but could cause issues.")
        return True
    else:
        print("❌ Python version mismatch. Please install Python 3.10.9")
        return False

def check_dependencies():
    print("\n=== Checking Dependencies ===")
    required_file = "requirements.txt"
    
    if not os.path.exists(required_file):
        print(f"❌ {required_file} not found!")
        return False
    
    with open(required_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    all_satisfied = True
    missing = []
    version_mismatch = []
    
    print(f"Checking {len(requirements)} required packages...")
    
    for req in requirements:
        package_name = req.split('==')[0] if '==' in req else req
        required_version = req.split('==')[1] if '==' in req else None
        
        try:
            installed_package = pkg_resources.get_distribution(package_name)
            installed_version = installed_package.version
            
            if required_version and installed_version != required_version:
                version_mismatch.append((package_name, installed_version, required_version))
                all_satisfied = False
            else:
                print(f"✅ {package_name} {'==' + installed_version if installed_version else ''}")
        except pkg_resources.DistributionNotFound:
            missing.append(package_name)
            all_satisfied = False
    
    if missing:
        print("\nMissing packages:")
        for package in missing:
            print(f"❌ {package}")
    
    if version_mismatch:
        print("\nVersion mismatches:")
        for package, installed, required in version_mismatch:
            print(f"⚠️ {package}: installed={installed}, required={required}")
    
    return all_satisfied

def check_streamlit():
    print("\n=== Checking Streamlit ===")
    try:
        import streamlit
        print(f"✅ Streamlit is installed (version {streamlit.__version__})")
        return True
    except ImportError:
        print("❌ Streamlit is not installed!")
        return False

def check_env_file():
    print("\n=== Checking Environment File ===")
    if os.path.exists(".env"):
        print("✅ .env file exists")
        
        with open(".env", 'r') as f:
            env_content = f.read()
        
        required_vars = ["SUPABASE_URL", "SUPABASE_KEY"]
        missing_vars = []
        
        for var in required_vars:
            if var not in env_content:
                missing_vars.append(var)
        
        if missing_vars:
            print("⚠️ Some required environment variables are missing:")
            for var in missing_vars:
                print(f"  - {var}")
            return False
        else:
            print("✅ All required environment variables are present")
            return True
    else:
        print("❌ .env file is missing!")
        return False

def main():
    print("=== Python 3.10.9 Compatibility Check ===")
    print("Checking if your environment is ready for Streamlit deployment...")
    
    python_check = check_python_version()
    dependencies_check = check_dependencies()
    streamlit_check = check_streamlit()
    env_check = check_env_file()
    
    print("\n=== Summary ===")
    print(f"Python version check: {'✅ Passed' if python_check else '❌ Failed'}")
    print(f"Dependencies check: {'✅ Passed' if dependencies_check else '❌ Failed'}")
    print(f"Streamlit check: {'✅ Passed' if streamlit_check else '❌ Failed'}")
    print(f"Environment file check: {'✅ Passed' if env_check else '❌ Failed'}")
    
    all_passed = python_check and dependencies_check and streamlit_check and env_check
    
    if all_passed:
        print("\n✅ All checks passed! Your environment is ready for Streamlit deployment.")
    else:
        print("\n❌ Some checks failed. Please fix the issues before deploying to Streamlit.")

if __name__ == "__main__":
    main()
