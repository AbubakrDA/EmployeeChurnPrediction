import sys
import os

# Add project directory to sys.path
sys.path.append(os.getcwd())

print("Checking imports...")

try:
    import employeechurnFastapi
    print("SUCCESS: employeechurnFastapi imported.")
except Exception as e:
    print(f"FAILURE: employeechurnFastapi import failed: {e}")

try:
    # Streamlit apps are scripts, but importing them checks syntax
    import EmployeeChurnPred
    print("SUCCESS: EmployeeChurnPred imported (syntax check).")
except Exception as e:
    # Streamlit specific commands might fail on import if run as script, but syntax errors will be caught
    print(f"NOTE: EmployeeChurnPred import result: {e}")

print("Import check complete.")
