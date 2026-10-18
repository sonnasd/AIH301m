def extract_mimic_cohort(data_dir="data/mimic_demo", output_csv="data/raw_sepsis_cohort.csv"):
    raise NotImplementedError(
        "Real Sepsis-3 trajectory extraction is not implemented. The downloaded demo "
        "subset lacks vital signs, labs and treatment tables. No synthetic fallback "
        "was written. Run main_pipeline.py explicitly for the synthetic coursework demo."
    )


if __name__ == "__main__":
    extract_mimic_cohort()
