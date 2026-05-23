from src.pipeline import run_full_pipeline

def main():
    print("=========================================")
    print("      Welcome to FinOpsLens Engine       ")
    print("=========================================\n")
    
    # Trigger the centralized execution pipeline
    run_full_pipeline()

if __name__ == "__main__":
    main()