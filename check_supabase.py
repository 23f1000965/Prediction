"""
Supabase dependency checker for Streamlit deployment
This script ensures that the correct Supabase package is installed
"""
import sys
import subprocess
import importlib.util

def check_supabase():
    """Check if the supabase package is properly installed"""
    print("Checking Supabase package...")
    
    # Try to import the supabase package
    try:
        import supabase
        from supabase import create_client
        print(f"✅ Supabase package is installed (using module: {supabase.__name__})")
        return True
    except ImportError:
        print("❌ Supabase package is not installed")
        
        # Try to install the package
        print("Attempting to install the Supabase package...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "supabase>=1.0.3"])
            print("✅ Supabase package installed successfully")
            
            # Verify installation
            import supabase
            from supabase import create_client
            print(f"✅ Verified Supabase package is now installed (using module: {supabase.__name__})")
            return True
        except Exception as e:
            print(f"❌ Failed to install Supabase package: {str(e)}")
            return False

if __name__ == "__main__":
    check_supabase()
