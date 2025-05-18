import streamlit.web.cli as stcli
import sys
import os

if __name__ == "__main__":
    # Get the absolute path to the app/app.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app", "app.py")
    
    # Check if file exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find {app_path}")
        sys.exit(1)
    
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main()) 