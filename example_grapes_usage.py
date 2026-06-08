#!/usr/bin/env python3
"""
GRAPES-SHAP Complete Example

This script demonstrates full end-to-end usage of the GRAPES-SHAP pipeline
with all 12 steps implemented according to the HTML architecture.

Usage:
    python example_grapes_usage.py

Requirements:
    - DeepSeek API key set via DEEPSEEK_API_KEY environment variable
    - All dependencies from requirements.txt installed
"""

import os
import json
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from grapes_shap.config import Config
from grapes_shap.inference.grapes_pipeline import GRAPESPipeline


def main():
    """Run complete GRAPES-SHAP inference example."""
    
    # ===== SETUP =====
    print("\n" + "="*70)
    print("GRAPES-SHAP Medical Reasoning System")
    print("="*70 + "\n")
    
    # Initialize configuration
    cfg = Config()
    
    # Check for DeepSeek API key
    if not cfg.deepseek_api_key:
        print("WARNING: DEEPSEEK_API_KEY not set!")
        print("To enable LLM features, set: export DEEPSEEK_API_KEY='sk-...'")
        print("Continuing with LLM fallback mode...\n")
    
    # Print configuration
    cfg.print_config()
    
    # ===== INITIALIZE PIPELINE =====
    print("Initializing GRAPES-SHAP pipeline...")
    pipeline = GRAPESPipeline(cfg)
    
    # ===== SAMPLE DATA =====
    # Example clinical case
    query = """
    70-year-old male with stage IIIB non-small cell lung cancer (NSCLC).
    EGFR exon 19 deletion confirmed by molecular testing.
    Performance status ECOG 1.
    Prior 4-cycle cisplatin/pemetrexed chemotherapy with progression.
    3 asymptomatic brain metastases noted on MRI.
    PD-L1 tumor proportion score 15%.
    What is the optimal treatment strategy?
    """
    
    # Example evidence documents (simplified)
    documents = [
        """FLAURA trial: Osimertinib demonstrated superior efficacy in EGFR-mutant NSCLC.
        Objective response rate 80.7%, median PFS 18.9 months vs 10.2 months for chemotherapy.
        1-year OS 83.3% vs 71.1%. First-generation EGFR TKI resistance can be overcome
        with osimertinib in patients with T790M emergence.""",
        
        """CNS penetration and intracranial efficacy of osimertinib in brain metastases.
        Intracranial objective response rate 91% in patients with brain metastases.
        Brain metastasis progression rate significantly lower than extracranial progression.
        Recommended for patients with EGFR-mutant NSCLC and CNS involvement.""",
        
        """Cisplatin resistance mechanisms in NSCLC: ERCC1 upregulation, Bcl-2 expression,
        and MDR1 pathway activation. Cross-resistance between platinum agents common.
        Alternative agents preferred after platinum failure. EGFR status-independent
        of platinum sensitivity in EGFR-mutant subsets.""",
        
        """Osimertinib adverse events profile: Grade ≥3 toxicity in 26% of patients.
        Most common: dermatitis (58%), diarrhea (41%), nausea (41%).
        Generally manageable with dose modifications. Elderly patients (>70 years):
        comparable safety to younger patients in real-world analysis.""",
        
        """EGFR T790M emergence as resistance mechanism after first-generation TKI.
        Third-generation TKIs (osimertinib) specifically designed to overcome T790M.
        50-60% of patients developing TKI resistance acquire T790M mutation.
        Superior CNS penetration vs first/second generation TKIs.""",
        
        """Real-world outcomes of osimertinib in EGFR-mutant NSCLC with brain metastases.
        1-year overall survival 78-85% depending on prognostic factors.
        Age >70 years not independently associated with worse prognosis.
        Median OS not reached at 24-month follow-up in subset with brain mets.""",
    ]
    
    print(f"\nClinical Query ({len(query)} characters):")
    print("-" * 60)
    print(query.strip())
    print()
    
    # ===== PIPELINE INFERENCE =====
    print("="*70)
    print("Running GRAPES-SHAP Inference Pipeline (12 steps)")
    print("="*70 + "\n")
    
    output = pipeline.infer(query, documents)
    
    # ===== RESULTS DISPLAY =====
    print("\n" + "="*70)
    print("RESULTS - Complete Analysis")
    print("="*70 + "\n")
    
    # Step 2: Query Expansion
    print("STEP 2: Query Expansion (HyDE + Sub-queries)")
    print("-" * 60)
    print(f"Original: {output.expanded_queries['original'][:80]}...")
    print(f"HyDE: {output.expanded_queries['hyde'][:80]}...")
    print(f"Sub-queries:")
    for i, sq in enumerate(output.expanded_queries['sub_queries'], 1):
        print(f"  {i}. {sq}")
    print()
    
    # Steps 3-4: Retrieval & Re-ranking
    print("STEPS 3-4: Retrieval & Re-ranking (Dense + BM25 + MMR)")
    print("-" * 60)
    print(f"Retrieved {len(output.retrieved_documents)} documents:")
    for i, (doc, score) in enumerate(zip(output.retrieved_documents, output.retrieval_scores), 1):
        print(f"\n  [{i}] Score: {score:.3f}")
        print(f"      {doc[:70]}...")
    print()
    
    # Step 5: Knowledge Graph
    print("STEP 5: Causal Knowledge Graph + GNN")
    print("-" * 60)
    if output.graph_embedding is not None:
        print(f"Graph embedding shape: {output.graph_embedding.shape}")
        print(f"Graph embedding summary: mean={output.graph_embedding.mean():.3f}, "
              f"std={output.graph_embedding.std():.3f}")
    print()
    
    # Steps 6-7: Planning
    print("STEPS 6-7: World Model Simulation + Tree-of-Thought Planning")
    print("-" * 60)
    print(f"Best plan score: {output.world_model_score:.3f}")
    if output.best_plan:
        print(f"Best treatment actions: {', '.join(output.best_plan[:3])}")
    print(f"Plan scores: {output.plan_scores}")
    print()
    
    # Step 8: Ensemble
    print("STEP 8: Deep Ensemble (5-model uncertainty)")
    print("-" * 60)
    print(f"Predictions:")
    for key, val in output.ensemble_predictions.items():
        print(f"  {key}: {val}")
    print(f"Uncertainty:")
    print(f"  Epistemic (need more data): {output.ensemble_uncertainty['epistemic']:.4f}")
    print(f"  Aleatoric (inherent randomness): {output.ensemble_uncertainty['aleatoric']:.4f}")
    print()
    
    # Step 9: LLM Reasoning
    print("STEP 9: LLM Chain-of-Thought Reasoning (DeepSeek)")
    print("-" * 60)
    print("Reasoning (first 500 chars):")
    print(output.llm_reasoning[:500] + "...\n")
    print(f"Recommendation:\n{output.llm_recommendation}\n")
    print(f"Confidence: {output.llm_confidence:.1%}")
    print()
    
    # Step 10: Hallucination Detection
    print("STEP 10: Hallucination Detection (Self-RAG + NLI)")
    print("-" * 60)
    halluc = output.hallucination_check
    print(f"Total claims analyzed: {halluc['nli']['total_claims']}")
    print(f"Supported claims: {halluc['nli']['supported_claims']}")
    print(f"Partial claims: {halluc['nli']['partial_claims']}")
    print(f"Grounding score: {halluc['nli']['grounding_score']:.1%}")
    print(f"Hallucination rate: {halluc['nli']['hallucination_rate']:.1%}")
    if halluc['nli']['flagged_claims']:
        print(f"Flagged claims: {halluc['nli']['flagged_claims'][:2]}")
    print(f"Recommended action: {halluc['recommended_action']}")
    print()
    
    # Step 11: SHAP Attribution
    print("STEP 11: SHAP Attribution (Shapley + Interactions)")
    print("-" * 60)
    shap_data = output.shap_values['interpretation']
    print("Top documents by Shapley value:")
    for doc_info in shap_data['top_documents'][:3]:
        print(f"\n  [{doc_info['index']+1}] φ = {doc_info['shapley_value']:+.3f}")
        print(f"      {doc_info['document'][:60]}...")
    
    if shap_data['synergies']:
        print(f"\nSynergies found:")
        for syn in shap_data['synergies'][:2]:
            print(f"  Docs [{syn['doc1_index']+1}] + [{syn['doc2_index']+1}]: "
                  f"I = +{syn['interaction_value']:.3f} (synergistic)")
    print()
    
    # Performance Metrics
    print("STEP 12: Performance Metrics")
    print("-" * 60)
    print("Timing breakdown:")
    for component, time_sec in sorted(output.timings.items()):
        pct = (time_sec / sum(output.timings.values())) * 100
        print(f"  {component:.<30} {time_sec:>6.3f}s ({pct:>5.1f}%)")
    total_time = sum(output.timings.values())
    print(f"  {'Total':.<30} {total_time:>6.3f}s (100.0%)")
    print()
    
    print("Quality scores:")
    for key, val in output.quality_scores.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.1%}")
    print()
    
    # ===== EXPORT RESULTS =====
    output_file = Path(__file__).parent / "outputs" / "grapes_example_output.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    export_data = {
        "query": output.query,
        "expanded_queries": output.expanded_queries,
        "retrieved_documents": output.retrieved_documents,
        "retrieval_scores": output.retrieval_scores,
        "world_model_score": output.world_model_score,
        "best_plan": output.best_plan,
        "ensemble_predictions": output.ensemble_predictions,
        "ensemble_uncertainty": output.ensemble_uncertainty,
        "llm_recommendation": output.llm_recommendation,
        "llm_confidence": output.llm_confidence,
        "hallucination_check": {
            "grounding_score": output.hallucination_check["nli"]["grounding_score"],
            "hallucination_rate": output.hallucination_check["nli"]["hallucination_rate"],
            "recommended_action": output.hallucination_check["recommended_action"]
        },
        "shap_summary": output.shap_values["interpretation"]["summary"],
        "timings": output.timings,
        "quality_scores": output.quality_scores
    }
    
    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✓ Results exported to: {output_file}")
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("GRAPES-SHAP Pipeline Complete")
    print("="*70)
    print(f"\n✓ All 12 steps executed successfully")
    print(f"✓ Total inference time: {total_time:.2f}s")
    print(f"✓ Recommendation confidence: {output.llm_confidence:.0%}")
    print(f"✓ Hallucination rate: {output.hallucination_check['nli']['hallucination_rate']:.1%}")
    print(f"✓ Grounding score: {output.hallucination_check['nli']['grounding_score']:.1%}")
    print()


if __name__ == "__main__":
    main()
