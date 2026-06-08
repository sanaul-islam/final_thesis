#!/usr/bin/env python3
"""
Complex Medical Prompts & Model Testing
Comprehensive test suite with realistic medical scenarios
Run: python scripts/test_complex_prompts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grapes_shap.config import CFG, SAVE_DIR, FIG_DIR


# Complex medical test cases
COMPLEX_PROMPTS = [
    {
        "id": 1,
        "category": "Acute Coronary Syndrome",
        "query": "A 58-year-old male with history of hypertension and diabetes presents to the ED with acute onset chest pain radiating to left arm and jaw. EKG shows ST elevation in leads II, III, aVF. Troponin is elevated at 2.5 ng/mL. Patient reports chest pain for 45 minutes. What is the most likely diagnosis and appropriate immediate management?",
        "expected_outcomes": ["Acute MI", "STEMI", "Immediate PCI/Thrombolysis"],
        "complexity": "High - Multiple risk factors, acute presentation"
    },
    {
        "id": 2,
        "category": "Sepsis & Infection",
        "query": "A 72-year-old female with COPD presents with fever (39.2C), altered mental status, and tachycardia (HR 118). Labs show WBC 15.2K, lactate 3.2 mmol/L, and creatinine increased from baseline. Imaging shows pneumonia. Blood cultures pending. Patient is hypotensive (BP 88/52). What is the first-line treatment approach?",
        "expected_outcomes": ["Sepsis", "Septic shock", "Broad-spectrum antibiotics"],
        "complexity": "High - Organ dysfunction, shock state"
    },
    {
        "id": 3,
        "category": "Metabolic Crisis",
        "query": "A 34-year-old Type 1 diabetic presents with polyuria, polydipsia, and dyspnea. pH is 7.15, HCO3 12, glucose 524 mg/dL, serum osmolarity 310 mOsm/kg, anion gap 18. Urinalysis shows large ketones. Patient consumed alcohol yesterday. What metabolic emergency is present?",
        "expected_outcomes": ["DKA", "Metabolic acidosis", "Insulin therapy"],
        "complexity": "High - Derangement of acid-base balance"
    },
    {
        "id": 4,
        "category": "Neurological Emergency",
        "query": "A 65-year-old right-handed male with atrial fibrillation (not anticoagulated) presents with acute onset aphasia and right-sided weakness (arm MRC 2/5, leg 3/5). Last known normal was 1.5 hours ago. NIHSS 14. CT head shows no hemorrhage. What is the time-sensitive intervention?",
        "expected_outcomes": ["Acute stroke", "tPA", "Thrombolytic therapy"],
        "complexity": "High - Narrow therapeutic window"
    },
    {
        "id": 5,
        "category": "Respiratory Failure",
        "query": "A 45-year-old smoker with COPD (FEV1 35%) presents with increased dyspnea, yellow sputum, and leg edema. ABG shows pH 7.28, pCO2 68, pO2 48 on RA. JVP elevated, crackles present, S3 gallop. CXR shows infiltrates. BNP 450. What constellation of problems is present?",
        "expected_outcomes": ["COPD exacerbation", "Cor pulmonale", "Heart failure"],
        "complexity": "Very High - Multiple system failure"
    },
    {
        "id": 6,
        "category": "Renal Crisis",
        "query": "A 52-year-old with poorly controlled hypertension and diabetes presents with severe headache, visual changes, and BP 210/130. Urinalysis shows 3+ protein and RBC casts. Creatinine 3.2 (baseline 1.0). Platelet count 85K. Schistocytes on blood smear. What hypertensive emergency syndrome is this?",
        "expected_outcomes": ["Hypertensive emergency", "TMA", "MAHA"],
        "complexity": "Very High - Multi-organ involvement"
    },
    {
        "id": 7,
        "category": "Gastrointestinal Bleeding",
        "query": "A 68-year-old on warfarin (INR 4.2) for AFib presents with hematemesis and melena. Vitals: HR 118, BP 92/58, RR 22. Hgb 7.2 (baseline 13). Patient is pale and diaphoretic. Already received 2 units PRBCs. Upper endoscopy shows spurting vessel in gastric antrum. What is the next immediate step?",
        "expected_outcomes": ["Variceal/nonvariceal bleeding", "Hemostasis needed", "Transfusion"],
        "complexity": "Very High - Hemorrhagic shock"
    },
    {
        "id": 8,
        "category": "Toxicology & Overdose",
        "query": "A 28-year-old found unresponsive with empty bottles of benzodiazepines and alcohol nearby. GCS 6, bradycardic (HR 42), hypotensive (BP 78/45), respiratory depression (RR 8). Pinpoint pupils present. Toxicology screen pending. What is the immediate management priority?",
        "expected_outcomes": ["Overdose", "CNS depression", "Airway protection"],
        "complexity": "Very High - Life-threatening intoxication"
    },
    {
        "id": 9,
        "category": "Autoimmune/Inflammatory",
        "query": "A 35-year-old female with SLE presents with pleuritic chest pain, dyspnea, and rash. Labs show low complement (C3 12, C4 8), elevated ANA 1:1280, anti-dsDNA positive, proteinuria 2.5g/24h, and mild thrombocytopenia (98K). Chest imaging shows pleural effusion. What lupus manifestations are present?",
        "expected_outcomes": ["SLE flare", "Lupus nephritis", "Serositis"],
        "complexity": "High - Multiple organ system involvement"
    },
    {
        "id": 10,
        "category": "Oncologic Emergency",
        "query": "A 52-year-old with newly diagnosed small cell lung cancer presents with weakness, confusion, and elevated sodium (Na 152). SIADH suspected. Serum osmolality 310, urine osmolality 750, urine sodium 180. Patient on minimal fluid. What is the underlying mechanism and treatment goal?",
        "expected_outcomes": ["SIADH", "Hypernatremia", "Fluid restriction"],
        "complexity": "Very High - Endocrine emergency in cancer"
    }
]


def print_prompt_analysis():
    """Print detailed analysis of test prompts."""
    
    print("\n" + "=" * 90)
    print("  COMPLEX MEDICAL PROMPTS & MODEL TESTING")
    print("=" * 90 + "\n")
    
    print("TEST SUITE OVERVIEW:")
    print("-" * 90)
    print(f"Total Prompts: {len(COMPLEX_PROMPTS)}")
    print(f"Categories: {len(set(p['category'] for p in COMPLEX_PROMPTS))}")
    print(f"Complexity Distribution:")
    
    complexity_dist = {}
    for p in COMPLEX_PROMPTS:
        comp = p['complexity'].split(' - ')[0]
        complexity_dist[comp] = complexity_dist.get(comp, 0) + 1
    
    for comp, count in sorted(complexity_dist.items()):
        print(f"  • {comp}: {count} prompts")
    
    print("\n" + "=" * 90)
    print("  DETAILED PROMPT ANALYSIS")
    print("=" * 90 + "\n")
    
    for prompt in COMPLEX_PROMPTS:
        print(f"\n[{prompt['id']}/10] {prompt['category'].upper()}")
        print("-" * 90)
        print(f"Complexity: {prompt['complexity']}")
        print(f"\nQuery:\n{prompt['query']}")
        print(f"\nExpected Model Outputs:")
        for i, outcome in enumerate(prompt['expected_outcomes'], 1):
            print(f"  {i}. {outcome}")
    
    print("\n" + "=" * 90)
    print("  TESTING FRAMEWORK")
    print("=" * 90 + "\n")
    
    print("EVALUATION CRITERIA:")
    print("-" * 90)
    criteria = {
        "1. Diagnostic Accuracy": "Does model correctly identify primary diagnosis?",
        "2. Differential Generation": "Are relevant differentials included in predictions?",
        "3. Evidence Integration": "Is retrieval-augmented generation effective?",
        "4. Risk Stratification": "Does model appropriately identify severity/urgency?",
        "5. Explainability": "Can SHAP attribution explain model decisions?",
        "6. Clinical Reasoning": "Are outputs clinically coherent and justified?"
    }
    
    for criterion, description in criteria.items():
        print(f"{criterion}")
        print(f"  └─ {description}")
    
    print("\n" + "=" * 90)
    print("  EXPECTED MODEL BEHAVIOR")
    print("=" * 90 + "\n")
    
    expected_behavior = {
        "Observation Processing": [
            "✓ Parse clinical text into structured features",
            "✓ Identify key symptoms, vitals, lab values",
            "✓ Extract temporal information (onset, progression)"
        ],
        "Feature Encoding": [
            "✓ Convert observations to 64-dim feature vectors",
            "✓ Maintain temporal sequences (8 timesteps)",
            "✓ Normalize and scale appropriately"
        ],
        "RAG Retrieval": [
            "✓ Find 6 most relevant documents from 30K index",
            "✓ Combine BM25 (keyword) + dense (semantic) search",
            "✓ Rank by relevance to query"
        ],
        "World Model": [
            "✓ Predict next observation state",
            "✓ Learn disease progression dynamics",
            "✓ Output confidence intervals (uncertainty)"
        ],
        "Ensemble Prediction": [
            "✓ Generate 5-dimensional outcome predictions",
            "✓ Provide probability estimates",
            "✓ Calibrate confidence"
        ],
        "SHAP Explanation": [
            "✓ Identify influential documents",
            "✓ Calculate feature importance",
            "✓ Explain prediction rationale"
        ]
    }
    
    for phase, behaviors in expected_behavior.items():
        print(f"{phase}:")
        for behavior in behaviors:
            print(f"  {behavior}")
    
    print("\n" + "=" * 90)
    print("  RESULTS INTERPRETATION GUIDE")
    print("=" * 90 + "\n")
    
    print("Loss Metrics (Training):")
    print("  • Total Loss: MSE + Smoothness + Uncertainty regularization")
    print("    ✓ Should decrease: Initial ~0.7 → Final ~0.05 (90% reduction)")
    print("  • Recon Loss: Next-step prediction accuracy")
    print("    ✓ Should decrease: Initial ~0.4 → Final ~0.02 (95% reduction)")
    print("  • Learning Rate: OneCycleLR schedule")
    print("    ✓ Ramp up → Peak → Ramp down for stable convergence")
    
    print("\nInference Metrics (Testing):")
    print("  • Retrieved Docs: Quality of RAG retrieval")
    print("    ✓ Top docs should be clinically relevant to query")
    print("  • Predictions: Output probability distributions")
    print("    ✓ Should be calibrated (well-matched to accuracy)")
    print("  • SHAP Values: Feature attribution scores")
    print("    ✓ High |SHAP| = influential for prediction")
    print("    ✓ Consistent with clinical judgment")
    
    print("\nModel Quality Indicators:")
    print("  ✓ Convergence: Training loss stabilizes by epoch 10-12")
    print("  ✓ Generalization: Val loss tracks training loss")
    print("  ✓ Calibration: Predicted probabilities match observed frequencies")
    print("  ✓ Explainability: SHAP attributions align with clinical reasoning")
    
    print("\n" + "=" * 90)
    print("  RUNNING MODEL ON TEST PROMPTS")
    print("=" * 90 + "\n")
    
    print("To execute inference on these prompts:")
    print("  1. Ensure models are trained: python scripts/train_world_model.py")
    print("  2. Run inference: python scripts/test_inference.py")
    print("  3. Review SHAP explanations: python scripts/visualize_shap.py")
    
    print("\nExpected Output:")
    print("  • Diagnosis predictions with confidence scores")
    print("  • Retrieved medical documents with relevance scores")
    print("  • SHAP feature importance visualization")
    print("  • Outcome predictions (prognosis, treatment response, risk)")
    
    print("\n" + "=" * 90)


def save_prompts_to_file():
    """Save prompts to a JSON file for later use."""
    import json
    
    prompts_path = SAVE_DIR / "test_prompts.json"
    
    with open(prompts_path, 'w') as f:
        json.dump(COMPLEX_PROMPTS, f, indent=2)
    
    print(f"\nTest prompts saved to: {prompts_path}")
    return True


def main():
    """Run complete analysis."""
    
    try:
        print_prompt_analysis()
        save_prompts_to_file()
        
        print("\n✓ Analysis complete!")
        print(f"\nNext steps:")
        print(f"  1. Train models: python scripts/train_world_model.py")
        print(f"  2. Create visualizations: python scripts/create_all_visualizations.py")
        print(f"  3. Run inference: python scripts/test_inference.py")
        print(f"  4. View results in outputs/figures/\n")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
