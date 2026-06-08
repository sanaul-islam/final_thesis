from datasets import load_dataset

class DatasetLoader:
    """
    Loads three zero-barrier HuggingFace datasets:
      • DDXPlus    — 1.3M patients, differential diagnosis, symptoms, antecedents
      • MedMCQA   — 194K clinical QA with explanations (reasoning corpus)
      • MedQA     — 12K USMLE multi-step clinical reasoning questions
    """

    @staticmethod
    def load_ddxplus(n_train=80_000, n_val=10_000, n_test=10_000):
        print("  Loading DDXPlus from HuggingFace (no credentials needed)...")
        ds = load_dataset("aai530-group6/ddxplus", trust_remote_code=True)
        train_raw = ds["train"].select(range(min(n_train, len(ds["train"]))))
        val_raw   = ds["validate"].select(range(min(n_val, len(ds["validate"]))))
        test_raw  = ds["test"].select(range(min(n_test, len(ds["test"]))))
        print(f"  DDXPlus — train:{len(train_raw)} val:{len(val_raw)} test:{len(test_raw)}")
        return train_raw, val_raw, test_raw

    @staticmethod
    def load_medmcqa(n=50_000):
        print("  Loading MedMCQA from HuggingFace...")
        ds = load_dataset("openlifescienceai/medmcqa", trust_remote_code=True)
        data = ds["train"].select(range(min(n, len(ds["train"]))))
        print(f"  MedMCQA — {len(data)} QA samples, 21 medical subjects")
        return data

    @staticmethod
    def load_medqa(n=1_000):
        print("  Loading MedQA-USMLE from HuggingFace...")
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", trust_remote_code=True)
        data = ds["test"].select(range(min(n, len(ds["test"]))))
        print(f"  MedQA — {len(data)} USMLE test questions")
        return data
