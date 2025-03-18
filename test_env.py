def test_environment():
    try:
        # Try importing required packages
        import ragen
        import verl
        # Try basic functionality (adjust based on what you need to verify)
        print("✓ ragen package is available")
        print("✓ verl package is available")
        
        # You can add more specific tests here, for example:
        # - Check if specific functions exist
        # - Verify version numbers
        # - Test basic operations
        
        return True
        
    except ImportError as e:
        print(f"✗ Import Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing environment configuration...")
    success = test_environment()
    
    if success:
        print("\nEnvironment is properly configured!")
    else:
        print("\nEnvironment configuration has issues!")