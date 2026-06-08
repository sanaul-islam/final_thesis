import numpy as np
from typing import List, Dict
from tqdm import tqdm
from grapes_shap.config import Config

class DDXPlusPreprocessor:
    """
    Converts raw DDXPlus records into trajectory tensors suitable
    for world model training.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.symptom_vocab: Dict[str, int] = {}
        self.antecedent_vocab: Dict[str, int] = {}
        self.pathology_vocab: Dict[str, int] = {}
        self.evidence_vocab: Dict[str, int] = {}
        self.fitted = False

    def fit(self, records):
        all_symptoms, all_antecedents, all_pathologies, all_evidences = set(), set(), set(), set()
        for r in records:
            if isinstance(r.get("SYMPTOMS", []), list):
                for s in r.get("SYMPTOMS", []):
                    all_symptoms.add(str(s))
            if isinstance(r.get("ANTECEDENTS", []), list):
                for a in r.get("ANTECEDENTS", []):
                    all_antecedents.add(str(a))
            if r.get("PATHOLOGY"):
                all_pathologies.add(str(r["PATHOLOGY"]))
            if isinstance(r.get("EVIDENCES", []), list):
                for e in r.get("EVIDENCES", []):
                    all_evidences.add(str(e))

        self.symptom_vocab    = {s: i for i, s in enumerate(sorted(all_symptoms))}
        self.antecedent_vocab = {a: i for i, a in enumerate(sorted(all_antecedents))}
        self.pathology_vocab  = {p: i for i, p in enumerate(sorted(all_pathologies))}
        self.evidence_vocab   = {e: i for i, e in enumerate(sorted(all_evidences))}
        self.fitted = True
        print(f"  Vocabulary — symptoms:{len(self.symptom_vocab)} "
              f"antecedents:{len(self.antecedent_vocab)} "
              f"pathologies:{len(self.pathology_vocab)} "
              f"evidences:{len(self.evidence_vocab)}")

    def _encode_patient(self, r) -> Dict[str, np.ndarray]:
        obs_dim = self.cfg.obs_dim
        obs = np.zeros(obs_dim, dtype=np.float32)

        # Features 0-1: age (normalised 0-1) and sex
        age = float(r.get("AGE", 40)) / 100.0
        sex = 1.0 if str(r.get("SEX", "M")).upper() == "M" else 0.0
        obs[0] = age
        obs[1] = sex

        # Features 2-33: symptom binary flags (32 dim)
        syms = r.get("SYMPTOMS", [])
        if isinstance(syms, list):
            for s in syms:
                idx = self.symptom_vocab.get(str(s), -1)
                if idx >= 0 and 2 + (idx % 32) < obs_dim:
                    obs[2 + (idx % 32)] = 1.0

        # Features 34-63: antecedent flags (30 dim)
        ants = r.get("ANTECEDENTS", [])
        if isinstance(ants, list):
            for a in ants:
                idx = self.antecedent_vocab.get(str(a), -1)
                if idx >= 0 and 34 + (idx % 30) < obs_dim:
                    obs[34 + (idx % 30)] = 1.0

        # Build a simulated trajectory by presenting evidence incrementally
        evs = r.get("EVIDENCES", [])
        if not isinstance(evs, list):
            evs = []
        T = self.cfg.seq_len
        obs_seq = np.zeros((T, obs_dim), dtype=np.float32)
        actions  = np.zeros(T, dtype=np.int64)
        obs_seq[0] = obs.copy()
        for t in range(1, T):
            if t - 1 < len(evs):
                ev_idx = self.evidence_vocab.get(str(evs[t-1]), 0)
                actions[t-1] = ev_idx % self.cfg.action_dim
                feat_slot = 2 + (ev_idx % 32)
                if feat_slot < obs_dim:
                    obs_seq[t] = obs_seq[t-1].copy()
                    obs_seq[t][feat_slot] = min(1.0, obs_seq[t-1][feat_slot] + 0.15)
            else:
                obs_seq[t] = obs_seq[t-1].copy()

        # Outcomes: differential diagnosis probabilities (up to 5 pathologies)
        diff_diag = r.get("DIFFERENTIAL_DIAGNOSIS", [])
        outcomes = np.zeros(self.cfg.n_outcomes, dtype=np.float32)
        if isinstance(diff_diag, list):
            for k, pair in enumerate(diff_diag[:self.cfg.n_outcomes]):
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    outcomes[k] = float(pair[1])
        # Normalise so probabilities sum to 1
        s = outcomes.sum()
        if s > 0:
            outcomes /= s
        else:
            outcomes[0] = 1.0

        return {"obs": obs_seq, "actions": actions,
                "next_obs": np.roll(obs_seq, -1, axis=0),
                "outcomes": outcomes,
                "pathology_id": self.pathology_vocab.get(str(r.get("PATHOLOGY", "")), 0)}

    def transform(self, records) -> List[Dict]:
        processed = []
        for r in tqdm(records, desc="  Encoding patients", leave=False):
            try:
                processed.append(self._encode_patient(r))
            except Exception:
                continue
        return processed


class MedMCQAPreprocessor:
    """
    Converts MedMCQA QA items into a knowledge base corpus for RAG retrieval.
    Each item becomes a retrievable document: question + correct answer + explanation.
    """
    @staticmethod
    def to_documents(records) -> List[str]:
        docs = []
        for r in records:
            q = r.get("question", "")
            opts = [r.get("opa",""), r.get("opb",""), r.get("opc",""), r.get("opd","")]
            correct_idx = r.get("cop", 0)
            if isinstance(correct_idx, int) and 0 <= correct_idx < 4:
                answer = opts[correct_idx]
            else:
                answer = opts[0]
            exp = r.get("exp", "") or ""
            subject = r.get("subject_name", "Medicine")
            doc = f"[{subject}] Q: {q} A: {answer}. {exp[:200]}"
            docs.append(doc.strip())
        return [d for d in docs if len(d) > 20]


class MedQAPreprocessor:
    """
    Converts MedQA USMLE items into evaluation queries + ground truth for
    benchmarking the GRAPES-SHAP pipeline on USMLE-style clinical vignettes.
    """
    @staticmethod
    def to_queries(records) -> List[Dict]:
        queries = []
        for r in records:
            q = r.get("question", "")
            opts = r.get("options", {})
            answer_key = r.get("answer_idx", r.get("answer", "A"))
            if isinstance(opts, dict):
                options = opts
            else:
                options = {"A": str(opts[0]) if opts else "", "B": "", "C": "", "D": ""}
            correct = options.get(str(answer_key), "")
            queries.append({"question": q, "options": options,
                            "answer_key": str(answer_key), "correct_answer": correct})
        return [q for q in queries if len(q["question"]) > 10]
