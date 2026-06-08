import ast
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from grapes_shap.config import Config


def _as_list(value):
    """Parse a DDXPlus field that may be a real list or a JSON/Python-literal
    string (the HuggingFace `aai530-group6/ddxplus` dataset stores
    DIFFERENTIAL_DIAGNOSIS and EVIDENCES as strings)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def _evidence_base(code: str) -> str:
    """Strip the value suffix from an evidence code, e.g.
    'E_54_@_V_161' -> 'E_54', 'E_56_@_4' -> 'E_56', 'E_48' -> 'E_48'."""
    return str(code).split("_@_")[0]


class DDXPlusPreprocessor:
    """
    Converts raw DDXPlus records into trajectory tensors suitable
    for world model training.

    The real HuggingFace DDXPlus schema exposes:
      AGE, SEX, PATHOLOGY, DIFFERENTIAL_DIAGNOSIS (str), EVIDENCES (str),
      INITIAL_EVIDENCE. Clinical signal comes from the EVIDENCES codes and
      the differential-diagnosis probability distribution.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pathology_vocab: Dict[str, int] = {}
        self.evidence_vocab: Dict[str, int] = {}
        self.fitted = False

    def fit(self, records):
        all_pathologies, all_evidences = set(), set()
        for r in records:
            if r.get("PATHOLOGY"):
                all_pathologies.add(str(r["PATHOLOGY"]))
            for e in _as_list(r.get("EVIDENCES")):
                all_evidences.add(_evidence_base(e))
            init_ev = r.get("INITIAL_EVIDENCE")
            if init_ev:
                all_evidences.add(_evidence_base(init_ev))

        self.pathology_vocab = {p: i for i, p in enumerate(sorted(all_pathologies))}
        self.evidence_vocab  = {e: i for i, e in enumerate(sorted(all_evidences))}
        self.fitted = True
        print(f"  Vocabulary — pathologies:{len(self.pathology_vocab)} "
              f"evidences:{len(self.evidence_vocab)}")

    def _encode_patient(self, r) -> Dict[str, np.ndarray]:
        obs_dim   = self.cfg.obs_dim
        feat_slots = obs_dim - 2          # slots reserved for evidence flags
        obs = np.zeros(obs_dim, dtype=np.float32)

        # Features 0-1: age (normalised 0-1) and sex
        age = float(r.get("AGE", 40)) / 100.0
        sex = 1.0 if str(r.get("SEX", "M")).upper() == "M" else 0.0
        obs[0] = age
        obs[1] = sex

        # Features 2..obs_dim-1: evidence binary flags
        evs = _as_list(r.get("EVIDENCES"))
        ev_indices = []
        for e in evs:
            idx = self.evidence_vocab.get(_evidence_base(e), -1)
            if idx >= 0:
                ev_indices.append(idx)
                obs[2 + (idx % feat_slots)] = 1.0

        # Build a simulated diagnostic trajectory: start from the initial
        # evidence then reveal additional evidence one step at a time.
        T = self.cfg.seq_len
        obs_seq = np.zeros((T, obs_dim), dtype=np.float32)
        actions = np.zeros(T, dtype=np.int64)

        base = np.zeros(obs_dim, dtype=np.float32)
        base[0] = age
        base[1] = sex
        init_idx = self.evidence_vocab.get(_evidence_base(r.get("INITIAL_EVIDENCE", "")), -1)
        if init_idx >= 0:
            base[2 + (init_idx % feat_slots)] = 1.0
        obs_seq[0] = base
        for t in range(1, T):
            obs_seq[t] = obs_seq[t-1].copy()
            if t - 1 < len(ev_indices):
                ev_idx = ev_indices[t-1]
                actions[t-1] = ev_idx % self.cfg.action_dim
                obs_seq[t][2 + (ev_idx % feat_slots)] = 1.0

        # Outcomes: top-k differential diagnosis probabilities (real distribution)
        diff_diag = _as_list(r.get("DIFFERENTIAL_DIAGNOSIS"))
        outcomes = np.zeros(self.cfg.n_outcomes, dtype=np.float32)
        top_names = []
        for k, pair in enumerate(diff_diag[:self.cfg.n_outcomes]):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                outcomes[k] = float(pair[1])
                top_names.append(str(pair[0]))
        s = outcomes.sum()
        if s > 0:
            outcomes /= s
        else:
            outcomes[0] = 1.0

        # Diagnosis class = position of the TRUE pathology within the top-k
        # differential (a meaningful, non-trivial classification target). If
        # the true pathology is not in the top-k, default to rank 0.
        pathology = str(r.get("PATHOLOGY", ""))
        diag_class = 0
        if pathology in top_names:
            diag_class = top_names.index(pathology)

        return {"obs": obs_seq, "actions": actions,
                "next_obs": np.roll(obs_seq, -1, axis=0),
                "outcomes": outcomes,
                "diag_class": diag_class,
                "pathology_id": self.pathology_vocab.get(pathology, 0)}

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
