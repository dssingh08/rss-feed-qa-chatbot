"""
Standalone Arize Phoenix Server Launcher
Run this script to start a central, persistent observability server that all your projects can send traces to.
"""

import os
import phoenix as px

if __name__ == "__main__":
    # Create a persistent directory in the project root for traces
    data_dir = os.path.join(os.getcwd(), ".phoenix_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Set the working directory environment variable for Phoenix
    os.environ["PHOENIX_WORKING_DIR"] = data_dir
    os.environ["PHOENIX_PORT"] = "6006"
    
    print("=" * 60)
    print(f"🚀 Starting Persistent Arize Phoenix Server")
    print(f"📁 Data Directory: {data_dir}")
    print(f"🌐 Dashboard: http://localhost:6006")
    print("=" * 60)
    print("Keep this terminal window open to preserve your trace data.")
    
    import subprocess
    import sys
    
    try:
        subprocess.run([sys.executable, "-m", "phoenix.server.main", "serve"], check=True)
    except KeyboardInterrupt:
        print("\nShutdown requested... exiting.")
    except Exception as e:
        print(f"\nFailed to start server: {e}")
