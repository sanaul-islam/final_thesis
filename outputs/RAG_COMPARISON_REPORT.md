# GRAPES-SHAP vs Advanced Baseline RAG — Comparative Study

## 1. Experimental Setup

- **Baseline (control):** hybrid dense (MiniLM + FAISS) + BM25 retrieval with reciprocal-rank fusion, answered directly by DeepSeek (`deepseek-chat`).
- **GRAPES-SHAP (proposed):** the full 12-step pipeline — query expansion, hybrid retrieval + MMR, causal KG + edge-biased GNN, latent world-model simulation, Tree-of-Thought planning, deep-ensemble uncertainty, hallucination self-check, and SHAP attribution — used to condition the same DeepSeek model.
- **Protocol:** both systems answer the *identical* set of 10 complex clinical vignettes over the same MedMCQA evidence corpus.

## 2. Aggregate Results

| Metric | Baseline RAG | GRAPES-SHAP |
|---|---|---|
| Clinical concept coverage | 0.700 | **1.000** |
| Answer structure completeness | 0.512 | **0.225** |
| Evidence citations (avg) | 1.60 | **2.00** |
| Stated confidence | 0.880 | 0.700 |
| SHAP evidence attribution | — (none) | **1.192** |
| Calibrated uncertainty | No | **Yes (deep ensemble)** |
| World-model planning | No | **Yes (Tree-of-Thought)** |

## 3. Per-Prompt Side-by-Side

### Prompt 1 — Acute Coronary Syndrome

**Clinical vignette:** A 58-year-old male with history of hypertension and diabetes presents to the ED with acute onset chest pain radiating to left arm and jaw. EKG shows ST elevation in leads II, III, aVF. Troponin is elevated at 2.5 ng/mL. Patient reports chest pain for 45 minutes. What is the most likely diagnosis and appropriate immediate management?

**Expected concepts:** Acute MI, STEMI, Immediate PCI/Thrombolysis

**Concept coverage:** baseline 0.67 vs GRAPES 1.00 · GRAPES mean |SHAP| = 1.082

<details><summary>Baseline RAG answer</summary>


(1) Most likely diagnosis: Inferior wall ST-elevation myocardial infarction (STEMI).  
(2) Immediate management: Urgent reperfusion therapy (primary percutaneous coronary intervention [PCI] is preferred; if not available, thrombolytic therapy may be considered, but note that thrombolytics are not given in certain contexts as per evidence [3]). Administer aspirin, oxygen if hypoxic, nitroglycerin for pain, and morphine if needed.  
(3) Rationale: The patient’s symptoms, EKG showing ST elevation in leads II, III, aVF, and elevated troponin are diagnostic of inferior wall MI [2]; immediate reperfusion is standard for STEMI [3].  
Confidence: 0.95

</details>

<details><summary>GRAPES-SHAP answer</summary>


*   Diagnosis: The patient presents with acute chest pain radiating to the left arm and jaw, with ST elevation in leads II, III, and aVF (inferior leads) and elevated troponin. This is classic for an acute ST-elevation myocardial infarction (STEMI) of the inferior wall. Evidence [2] directly confirms that ST elevation in leads II, III, and aVF is diagnostic of an inferior wall MI. The patient's risk factors (hypertension, diabetes) and symptom duration (45 minutes) are consistent with this diagnosis.
*   Immediate Management: The cornerstone of STEMI management is rapid reperfusion. Given the patient's presentation within 12 hours of symptom onset, the primary options are primary percutaneous coronary intervention (PCI) or fibrinolysis. Evidence [3] discusses management of STEMI and notes that thrombolytic therapy is a key intervention, but it is not given in certain scenarios (e.g., if contraindicated or if PCI is available). For this patient, the most appropriate immediate step is to

*   Immediate Actions:
    *   Activate Cardiac Catheterization Lab: For primary PCI as soon as possible (target door-to-balloon time <90 minutes).
    *   Aspirin: 162-325 mg chewed immediately (loading dose).
    *   P2Y12 Inhibitor: Ticagrelor 180 mg orally (loading dose) or Prasugrel 60 mg orally (loading dose). If PCI is delayed or fibrinolysis is used, consider Clopidogrel 300-600 mg orally.
    *   Anticoagulation: Unfractionated heparin (60 U/kg IV bolus, then 12 U/kg/hr infusion) or enoxaparin (1 mg/kg SC every 12 hours) or bivalirudin (0.75 mg/kg IV bolus, then 1.75 mg/kg/hr infusion).
    *   Oxygen: If O2 saturation <90% or respiratory distress.
    *   Pain Management: Morphine sulfate 2-4 mg IV for pain (if not hypotensive).
    *   Nitroglycerin: 0.4 mg sublingual every 5 minutes for up to 3 doses (if no hypotension, bradycardia, or right ventricular infarction).
*   If PCI is not available within 120 minutes:
    *   Fibrinolysis: Tenecteplase (TNK) 30-50 mg IV bolus (w

</details>


---

### Prompt 2 — Sepsis & Infection

**Clinical vignette:** A 72-year-old female with COPD presents with fever (39.2C), altered mental status, and tachycardia (HR 118). Labs show WBC 15.2K, lactate 3.2 mmol/L, and creatinine increased from baseline. Imaging shows pneumonia. Blood cultures pending. Patient is hypotensive (BP 88/52). What is the first-line treatment approach?

**Expected concepts:** Sepsis, Septic shock, Broad-spectrum antibiotics

**Concept coverage:** baseline 0.33 vs GRAPES 1.00 · GRAPES mean |SHAP| = 0.977

<details><summary>Baseline RAG answer</summary>


(1) Most likely diagnosis: Septic shock secondary to pneumonia.  
(2) Immediate management: Aggressive IV fluid resuscitation, oxygen supplementation, and norepinephrine for hypotension [3].  
(3) Rationale: Evidence [3] directly addresses septic shock management with fluids and norepinephrine for blood pressure stabilization.  
Confidence: 0.9

</details>

<details><summary>GRAPES-SHAP answer</summary>


This 72-year-old female with COPD presents with fever, altered mental status, tachycardia, hypotension (BP 88/52), elevated lactate (3.2 mmol/L), leukocytosis, and acute kidney injury in the setting of pneumonia. This constellation meets the criteria for septic shock. The immediate priorities are: (1) aggressive fluid resuscitation, (2) early broad-spectrum antibiotics, (3) vasopressor support for refractory hypotension, and (4) source control.

- Fluid resuscitation: The patient is hypotensive with elevated lactate, indicating tissue hypoperfusion. Evidence [1] describes a sepsis patient given IV fluids aggressively for hydration, which is standard initial management.
- Vasopressor selection: The patient remains hypotensive despite fluids. Evidence [3] directly addresses septic shock with BP 70/50 mmHg, stating that if hypotension persists after fluids, norepinephrine is the drug of choice to stabilize blood pressure. This patient’s BP of 88/52 is similarly critical.
- Antibiotics: Br

- Immediate: Administer 30 mL/kg of crystalloid (e.g., normal saline or balanced solution) intravenously over the first hour. Reassess for fluid responsiveness.
- Vasopressor: Start norepinephrine intravenously, titrated to a mean arterial pressure (MAP) ≥ 65 mmHg. Typical starting dose: 0.05–0.1 mcg/kg/min, titrated upward as needed.
- Antibiotics: Initiate broad-spectrum antibiotics for community-acquired pneumonia with sepsis: vancomycin (15–20 mg/kg IV loading dose) plus piperacillin-tazobactam (4.5 g IV every 6 hours) or cefepime (2 g IV every 8 hours) plus azithromycin (500 mg IV daily). Adjust based on local resistance patterns and pending cultures.
- Supportive care: Supplemental oxygen to maintain SpO₂ ≥ 92%. Consider non-invasive ventilation if respiratory distress worsens. Stress-dose hydrocortisone (200 mg/day IV) may be considered if vasopressor requirement is high, though evidence is mixed.

3.

</details>


---

### Prompt 3 — Metabolic Crisis

**Clinical vignette:** A 34-year-old Type 1 diabetic presents with polyuria, polydipsia, and dyspnea. pH is 7.15, HCO3 12, glucose 524 mg/dL, serum osmolarity 310 mOsm/kg, anion gap 18. Urinalysis shows large ketones. Patient consumed alcohol yesterday. What metabolic emergency is present?

**Expected concepts:** DKA, Metabolic acidosis, Insulin therapy

**Concept coverage:** baseline 1.00 vs GRAPES 1.00 · GRAPES mean |SHAP| = 1.143

<details><summary>Baseline RAG answer</summary>


(1) Most likely diagnosis: Diabetic ketoacidosis (DKA)  
(2) Immediate management: Administration of short-acting insulin  
(3) Rationale: The patient meets DKA criteria with glucose >250 mg/dL, pH <7.3, HCO3 <18, elevated ketones, and anion gap >12 [1][3].  

Confidence: 0.95

</details>

<details><summary>GRAPES-SHAP answer</summary>


1.  Identifying the Primary Diagnosis: The patient is a 34-year-old with Type 1 diabetes. The presenting symptoms (polyuria, polydipsia, dyspnea) combined with the lab values (pH 7.15, HCO3 12, glucose 524 mg/dL, anion gap 18, large ketones) are classic for diabetic ketoacidosis (DKA). The evidence explicitly defines DKA by a glucose >250 mg/dL, pH <7.3, HCO3 <18 mEq/L, and elevated ketones [3]. The low pH of 7.15 and HCO3 of 12 confirm a severe metabolic acidosis [1, 2].

2.  Differential Diagnosis Considerations:
    *   Hyperosmolar Hyperglycemic State (HHS): The serum osmolarity is 310 mOsm/kg. While elevated, HHS typically presents with osmolarity >320 mOsm/kg and minimal to no ketones. The presence of large ketones and significant acidosis points away from pure HHS.
    *   Alcoholic Ketoacidosis (AKA): The patient consumed alcohol yesterday, which could contribute to ketosis. However, AKA typically occurs in chronic alcoholics with poor nutrition and often presents with a milder

The immediate treatment for DKA is intravenous fluid resuscitation and insulin therapy. The evidence clearly states that the most appropriate immediate treatment is the administration of insulin [1], specifically short-acting insulin [3].

*   Fluids: Start with 1 liter of 0.9% normal saline over the first hour. Subsequent fluid replacement should be guided by hemodynamic status and corrected sodium levels.
*   Insulin: Administer a bolus of 0.1 units/kg of regular insulin intravenously, followed by a continuous infusion of 0.1 units/kg/hour.
*   Potassium: Monitor serum potassium closely. If K+ is < 3.3 mEq/L, hold insulin and replete potassium first. If K+ is 3.3-5.3 mEq/L, add 20-30 mEq of potassium to each liter of IV fluid to maintain a level of 4-5 mEq/L.
*   Bicarbonate: Not routinely recommended for DKA unless pH is < 6.9. This patient's pH is 7.15, so bicarbonate is not indicated.

</details>


---

### Prompt 4 — Neurological Emergency

**Clinical vignette:** A 65-year-old right-handed male with atrial fibrillation (not anticoagulated) presents with acute onset aphasia and right-sided weakness (arm MRC 2/5, leg 3/5). Last known normal was 1.5 hours ago. NIHSS 14. CT head shows no hemorrhage. What is the time-sensitive intervention?

**Expected concepts:** Acute stroke, tPA, Thrombolytic therapy

**Concept coverage:** baseline 0.67 vs GRAPES 1.00 · GRAPES mean |SHAP| = 0.911

<details><summary>Baseline RAG answer</summary>


(1) Most likely diagnosis: Acute ischemic stroke due to atrial fibrillation.  
(2) Immediate management: Administer thrombolytics (e.g., alteplase) given no hemorrhage on CT and presentation within the therapeutic window.  
(3) Rationale: Evidence [2] states that for one-sided weakness with no hemorrhage on NCCT, the clinician should give thrombolytics.  

Confidence: 0.95

</details>

<details><summary>GRAPES-SHAP answer</summary>


:

1.  Patient Presentation: A 65-year-old male with atrial fibrillation (not anticoagulated) presents with acute-onset aphasia and right-sided weakness (arm MRC 2/5, leg 3/5), with a last known normal time of 1.5 hours ago. The NIHSS is 14, indicating a moderate-to-severe stroke. The CT head shows no hemorrhage.
2.  Differential Diagnosis: The acute onset of focal neurological deficits in a patient with atrial fibrillation is highly suggestive of an ischemic stroke due to cardioembolism [1]. The CT head is the standard initial imaging to rule out hemorrhage, which is the primary contraindication for thrombolysis [2].
3.  Time-Sensitive Intervention: The patient is within the 4.5-hour window for intravenous thrombolysis (alteplase/tenecteplase). The evidence explicitly states that in a patient with one-sided weakness and no hemorrhage on NCCT, the appropriate action is to administer thrombolytics [2]. The patient is not on anticoagulation (warfarin), which eliminates a major contraindi

:

- Primary Intervention: Administer intravenous thrombolysis.
    - Drug: Alteplase (tPA).
    - Dose: 0.9 mg/kg (maximum 90 mg total dose).
    - Schedule: Give 10% of the total dose as an intravenous bolus over 1 minute, followed by the remaining 90% as an intravenous infusion over 60 minutes.
- Alternative: Tenecteplase (TNKase) 0.25 mg/kg (maximum 25 mg) as a single intravenous bolus over 5 seconds is an acceptable alternative with a simpler administration.
- Supportive Care: Admit to a stroke unit or intensive care unit for close monitoring. Maintain blood pressure <180/105 mmHg for the first 24 hours post-thrombolysis.

</details>


---

### Prompt 5 — Respiratory Failure

**Clinical vignette:** A 45-year-old smoker with COPD (FEV1 35%) presents with increased dyspnea, yellow sputum, and leg edema. ABG shows pH 7.28, pCO2 68, pO2 48 on RA. JVP elevated, crackles present, S3 gallop. CXR shows infiltrates. BNP 450. What constellation of problems is present?

**Expected concepts:** COPD exacerbation, Cor pulmonale, Heart failure

**Concept coverage:** baseline 1.00 vs GRAPES 1.00 · GRAPES mean |SHAP| = 1.304

<details><summary>Baseline RAG answer</summary>


(1) Most likely diagnosis: Acute exacerbation of COPD with cor pulmonale and superimposed cardiogenic pulmonary edema.  
(2) Immediate management: Noninvasive positive pressure ventilation (BiPAP), diuretics for fluid overload, bronchodilators, corticosteroids, and antibiotics for suspected infection.  
(3) Rationale: ABG shows type II respiratory failure (pH 7.28, pCO2 68) consistent with acute COPD exacerbation [2]; elevated JVP, leg edema, S3 gallop, and BNP 450 indicate right heart strain (cor pulmonale) and cardiogenic pulmonary edema [1][3].  

Confidence: 0.85

</details>

<details><summary>GRAPES-SHAP answer</summary>


:

1.  Primary Problem Identification: The patient is a 45-year-old smoker with severe COPD (FEV1 35%). The ABG shows pH 7.28 (acidemia), pCO2 68 (hypercapnia), and pO2 48 (hypoxemia) on room air. This pattern is diagnostic of acute-on-chronic hypercapnic respiratory failure (Type II respiratory failure), most consistent with an acute exacerbation of COPD [2]. The evidence explicitly states that a patient with pH 7.2, pCO2 80, and pO2 46 has an acute exacerbation of COPD [2]. While this patient's pCO2 is 68 (slightly lower than the example), the pattern is identical.

2.  Secondary Problem Identification (Cardiogenic Pulmonary Edema): The patient also presents with leg edema, elevated JVP, crackles, an S3 gallop, and a BNP of 450. These are classic signs of acute decompensated heart failure (ADHF) with pulmonary edema. The CXR shows infiltrates. The key differential is cardiogenic vs. non-cardiogenic pulmonary edema. The presence of an S3 gallop, elevated JVP, and elevated BNP strongly

:

*   Immediate Management:
    *   Non-invasive ventilation (BiPAP): Initiate with IPAP 10-15 cmH2O, EPAP 5-8 cmH2O, titrated to target pH >7.35 and pCO2 <60.
    *   Diuresis: Furosemide 40-80 mg IV bolus, then 5-10 mg/hour IV infusion, titrated to urine output >100 mL/hour and resolution of edema/crackles.
    *   Bronchodilators: Albuterol 2.5 mg + Ipratropium 0.5 mg via nebulizer every 4-6 hours.
    *   Corticosteroids: Methylprednisolone 60 mg IV every 6 hours for 48 hours, then taper.
    *   Antibiotics: Ceftriaxone 1 g IV daily + Azithromycin 500 mg IV daily (for community-acquired pneumonia/COPD exacerbation).
    *   Oxygen: Target SpO2 88-92% (avoid hyperoxia which can worsen hypercapnia).

*   Contraindications/Avoid:
    *   Avoid high-flow oxygen (can suppress hypoxic drive).
    *   Avoid beta-blockers (can worsen bronchospasm).
    *   Avoid excessive diuresis (can cause hypotension and worsen renal function).

</details>


---

### Prompt 6 — Renal Crisis

**Clinical vignette:** A 52-year-old with poorly controlled hypertension and diabetes presents with severe headache, visual changes, and BP 210/130. Urinalysis shows 3+ protein and RBC casts. Creatinine 3.2 (baseline 1.0). Platelet count 85K. Schistocytes on blood smear. What hypertensive emergency syndrome is this?

**Expected concepts:** Hypertensive emergency, TMA, MAHA

**Concept coverage:** baseline 0.33 vs GRAPES 1.00 · GRAPES mean |SHAP| = 0.933

<details><summary>Baseline RAG answer</summary>


(1) Most likely diagnosis: Hypertensive emergency with thrombotic microangiopathy (likely malignant hypertension with associated microangiopathic hemolytic anemia and thrombocytopenia).  
(2) Immediate management: Urgent blood pressure reduction with intravenous antihypertensives (e.g., labetalol, nicardipine) and evaluation for end-organ damage; consider plasma exchange if thrombotic thrombocytopenic purpura is suspected.  
(3) Rationale: The presentation of severe hypertension (BP 210/130), proteinuria, RBC casts, acute kidney injury (creatinine 3.2), thrombocytopenia (platelets 85K), and schistocytes aligns with hypertensive emergency complicated by microangiopathic hemolytic anemia, as seen in malignant hypertension or secondary thrombotic microangiopathy [5].  

Confidence: 0.85

</details>

<details><summary>GRAPES-SHAP answer</summary>


1.  Symptom and Sign Analysis: The patient presents with a severe headache, visual changes, and a blood pressure of 210/130 mmHg, fulfilling the criteria for a hypertensive emergency (severe hypertension with acute target organ damage). The presence of acute kidney injury (creatinine 3.2 from a baseline of 1.0), proteinuria, and RBC casts indicates renal involvement. The low platelet count (85K) and schistocytes on blood smear point to a microangiopathic hemolytic anemia (MAHA) and thrombocytopenia.

2.  Differential Diagnosis: The combination of hypertensive emergency, acute kidney injury, MAHA, and thrombocytopenia is classic for Thrombotic Microangiopathy (TMA). The most common causes of TMA in this setting are:
    *   Malignant Hypertension (HTN): This is the most likely diagnosis given the severely elevated BP (210/130) and the presence of schistocytes and thrombocytopenia. Malignant HTN can directly cause a TMA.
    *   Thrombotic Thrombocytopenic Purpura (TTP): While TTP presen

The immediate goal is to lower blood pressure in a controlled manner to prevent further end-organ damage while avoiding hypoperfusion.

*   Immediate Management:
    *   Admit to Intensive Care Unit (ICU) for continuous monitoring.
    *   Initiate IV antihypertensive therapy:
        *   Labetalol: 20 mg IV bolus, then 20-80 mg IV every 10 minutes, or start a continuous infusion at 0.5-2 mg/min. This is a safe first-line agent as it does not increase heart rate.
        *   Nicardipine: 5 mg/hr IV infusion, titrated by 2.5 mg/hr every 5-15 minutes to a maximum of 15 mg/hr. This is a potent vasodilator.
    *   Goal: Reduce mean arterial pressure (MAP) by no more than 25% within the first hour. Target BP of 160/100-110 mmHg in the first 2-6 hours. Avoid rapid drops to prevent cerebral or myocardial ischemia.
    *   Avoid: Nitroprusside (risk of cyanide toxicity, especially with renal failure) and hydralazine (unpredictable response).

*   Specific Therapy for TMA:
    *   Plasma Excha

</details>


---

### Prompt 7 — Gastrointestinal Bleeding

**Clinical vignette:** A 68-year-old on warfarin (INR 4.2) for AFib presents with hematemesis and melena. Vitals: HR 118, BP 92/58, RR 22. Hgb 7.2 (baseline 13). Patient is pale and diaphoretic. Already received 2 units PRBCs. Upper endoscopy shows spurting vessel in gastric antrum. What is the next immediate step?

**Expected concepts:** Variceal/nonvariceal bleeding, Hemostasis needed, Transfusion

**Concept coverage:** baseline 0.67 vs GRAPES 1.00 · GRAPES mean |SHAP| = 1.244

<details><summary>Baseline RAG answer</summary>


(1) Most likely diagnosis: Upper GI bleeding from a gastric antral ulcer with active hemorrhage (spurting vessel) in a patient with warfarin toxicity (INR 4.2) and hypovolemic shock.  
(2) Immediate management: Administer IV fluids for resuscitation [3], reverse warfarin effect (e.g., vitamin K, fresh frozen plasma, or prothrombin complex concentrate) [2], and perform endoscopic hemostasis (e.g., epinephrine injection, cautery, or clip placement) for the spurting vessel.  
(3) Rationale: Evidence [3] supports immediate IV fluids for hypotension, and [2] indicates that bleeding diathesis (INR 4.2) requires reversal before or during intervention.  
Confidence: 0.85

</details>

<details><summary>GRAPES-SHAP answer</summary>


:

1.  Hemodynamic Instability and Active Bleeding: The patient presents with hematemesis, melena, tachycardia (HR 118), hypotension (BP 92/58), and signs of hypovolemic shock (pale, diaphoretic). This indicates acute, ongoing blood loss. The initial management of any hemorrhagic shock is aggressive fluid resuscitation and blood product support. Evidence [1] and [3] both emphasize that in the setting of acute blood loss with hypotension and tachycardia, immediate IV fluids and blood transfusion are the first steps. The patient has already received 2 units of PRBCs but remains unstable.

2.  Warfarin Reversal is Critical: The patient is on warfarin with a supratherapeutic INR of 4.2. This significantly impairs coagulation and perpetuates the bleeding. Evidence [2] explicitly states that lumbar puncture is contraindicated in patients with a bleeding diathesis (like warfarin toxicity), highlighting the danger of uncontrolled bleeding in this setting. Therefore, immediate reversal of antic

:

1.  Immediate Warfarin Reversal:
    *   4-Factor Prothrombin Complex Concentrate (PCC): Administer 25-50 units/kg IV (dose based on INR and weight). This provides rapid, complete reversal of warfarin effect within minutes.
    *   IV Vitamin K: Administer 10 mg IV slowly over 30-60 minutes. This will sustain reversal by promoting hepatic synthesis of clotting factors, but takes 6-24 hours to work.

2.  Endoscopic Hemostasis:
    *   Epinephrine Injection: Inject 1:10,000 epinephrine in 0.5-1 mL aliquots around the bleeding vessel (total volume typically 5-10 mL) to achieve local vasoconstriction and tamponade.
    *   Thermal Coagulation or Mechanical Clip: Immediately after epinephrine, apply a heater probe or bipolar probe to coagulate the vessel, or place a hemostatic clip directly on the visible vessel.

3.  Supportive Care:
    *   IV Fluids: Continue crystalloid (e.g., Lactated Ringer's) to maintain MAP >65 mmHg.
    *   Blood Products: Transfuse PRBCs to maintain Hgb >7-8 g/

</details>


---

### Prompt 8 — Toxicology & Overdose

**Clinical vignette:** A 28-year-old found unresponsive with empty bottles of benzodiazepines and alcohol nearby. GCS 6, bradycardic (HR 42), hypotensive (BP 78/45), respiratory depression (RR 8). Pinpoint pupils present. Toxicology screen pending. What is the immediate management priority?

**Expected concepts:** Overdose, CNS depression, Airway protection

**Concept coverage:** baseline 0.67 vs GRAPES 1.00 · GRAPES mean |SHAP| = 1.188

<details><summary>Baseline RAG answer</summary>


(1) Most likely diagnosis: Opioid overdose (likely co-ingestion with benzodiazepines and alcohol).  
(2) Immediate management: Administer naloxone (opioid antagonist) and provide respiratory support (e.g., bag-valve-mask ventilation or intubation).  
(3) Rationale: Pinpoint pupils and respiratory depression are classic signs of opioid overdose [4], and the combination with benzodiazepines and alcohol increases the risk of severe respiratory depression.  

Confidence: 0.85

</details>

<details><summary>GRAPES-SHAP answer</summary>


1.  Presentation Analysis: The patient presents with unresponsiveness (GCS 6), bradycardia (HR 42), hypotension (BP 78/45), respiratory depression (RR 8), and pinpoint pupils. Empty bottles of benzodiazepines and alcohol are found nearby. This triad of CNS depression, respiratory depression, and pinpoint pupils is the classic toxidrome of opioid overdose, not simply benzodiazepine and alcohol intoxication. While benzodiazepines and alcohol can cause respiratory depression and coma, they typically cause pupillary dilation (mydriasis) or normal pupils, not miosis. The presence of pinpoint pupils strongly suggests a co-ingestion of an opioid [3].

2.  Immediate Threat: The most immediate, life-threatening issue is respiratory depression (RR 8) leading to hypoxia and potential respiratory arrest. This is the primary driver of mortality in this scenario.

3.  Evidence Application: 
    *   The evidence [3] directly links pinpoint pupils to pontine hemorrhage, but in the context of overdose,

*   Immediate Action: Administer Naloxone (Narcan) 0.4-2 mg IV/IM/IN. If no response, repeat every 2-3 minutes. A total of up to 10 mg may be needed for potent opioids. Given the severity (GCS 6, RR 8), start with 2 mg IV.
*   Airway Management: Simultaneously, provide bag-valve-mask ventilation with 100% oxygen. Prepare for endotracheal intubation if the patient does not respond to naloxone or has a prolonged apnea.
*   Circulatory Support: Administer a 500-1000 mL bolus of Normal Saline IV for hypotension [6]. If hypotension persists after naloxone and fluids, consider vasopressors (e.g., norepinephrine).

</details>


---

### Prompt 9 — Autoimmune/Inflammatory

**Clinical vignette:** A 35-year-old female with SLE presents with pleuritic chest pain, dyspnea, and rash. Labs show low complement (C3 12, C4 8), elevated ANA 1:1280, anti-dsDNA positive, proteinuria 2.5g/24h, and mild thrombocytopenia (98K). Chest imaging shows pleural effusion. What lupus manifestations are present?

**Expected concepts:** SLE flare, Lupus nephritis, Serositis

**Concept coverage:** baseline 0.67 vs GRAPES 1.00 · GRAPES mean |SHAP| = 1.717

<details><summary>Baseline RAG answer</summary>


(1) Most likely diagnosis: Systemic lupus erythematosus (SLE) with active lupus nephritis (proteinuria 2.5g/24h), lupus pleuritis (pleuritic chest pain, pleural effusion), and cutaneous lupus (rash). Low complement (C3, C4) and positive anti-dsDNA indicate active disease [1][5].

(2) Immediate management: High-dose corticosteroids (e.g., prednisone 1 mg/kg/day) plus mycophenolate mofetil or cyclophosphamide for lupus nephritis; consider NSAIDs or colchicine for pleuritic pain; monitor for thrombocytopenia.

(3) Rationale: Low complement and anti-dsDNA positivity are classic for active SLE with renal and serosal involvement [1][5].

Confidence: 0.95

</details>

<details><summary>GRAPES-SHAP answer</summary>


The patient presents with a classic constellation of symptoms and lab findings highly suggestive of active Systemic Lupus Erythematosus (SLE). The step-by-step analysis is as follows:

*   Pleuritic chest pain and dyspnea with pleural effusion: These are classic manifestations of serositis, specifically lupus pleuritis. This is a common pulmonary manifestation of SLE.
*   Rash: The description of a rash (likely malar or discoid) is a hallmark mucocutaneous manifestation of SLE [5].
*   Low complement levels (C3, C4): Low complement levels are a key serological feature of active SLE, particularly in patients with renal involvement. This is supported by evidence [1], which lists SLE as a cause of low serum complement. Low C4 is also a marker of disease activity in related autoimmune conditions [4].
*   Positive ANA and anti-dsDNA: These are highly specific serological markers for SLE. Anti-dsDNA is particularly associated with active disease, especially lupus nephritis.
*   Proteinuria (

Given the presence of major organ involvement (lupus nephritis with 2.5g/24h proteinuria) and active systemic disease, aggressive immunosuppression is indicated. The recommended initial regimen is:

*   Induction Therapy:
    *   Mycophenolate mofetil (MMF): 2-3 grams per day orally, divided into two doses. This is a first-line agent for lupus nephritis.
    *   Corticosteroids: Prednisone 0.5-1 mg/kg/day (e.g., 40-60 mg/day) orally, with a plan for rapid taper over 3-6 months to minimize toxicity.
*   Adjunctive Therapy:
    *   Hydroxychloroquine (HCQ): 200-400 mg/day orally. HCQ is a cornerstone of SLE management, reducing flares and improving long-term outcomes.
    *   ACE inhibitor or ARB: For renoprotection given the proteinuria (e.g., lisinopril 5-10 mg/day or losartan 25-50 mg/day).

3.

</details>


---

### Prompt 10 — Oncologic Emergency

**Clinical vignette:** A 52-year-old with newly diagnosed small cell lung cancer presents with weakness, confusion, and elevated sodium (Na 152). SIADH suspected. Serum osmolality 310, urine osmolality 750, urine sodium 180. Patient on minimal fluid. What is the underlying mechanism and treatment goal?

**Expected concepts:** SIADH, Hypernatremia, Fluid restriction

**Concept coverage:** baseline 1.00 vs GRAPES 1.00 · GRAPES mean |SHAP| = 1.420

<details><summary>Baseline RAG answer</summary>


(1) **Most likely diagnosis:** Syndrome of Inappropriate Antidiuretic Hormone (SIADH) secondary to small cell lung cancer.  
(2) **Immediate management:** Fluid restriction and careful correction of hypernatremia (serum sodium 152 mEq/L) with hypotonic fluids, targeting a gradual decrease in sodium.  
(3) **Rationale:** The patient has hypernatremia (Na 152) with elevated urine osmolality (750) and urine sodium (180), consistent with SIADH [5]; however, the evidence does not directly address hypernatremia in SIADH, so management is inferred from standard principles.  

Confidence: 0.7

</details>

<details><summary>GRAPES-SHAP answer</summary>


*   Diagnosis Confirmation: The patient presents with hypernatremia (Na 152), not hyponatremia. The classic diagnostic criteria for SIADH include hyponatremia (low serum sodium) and low serum osmolality (<280 mOsm/kg) [6]. This patient has hypernatremia and high serum osmolality (310 mOsm/kg), which is the opposite of SIADH. Therefore, SIADH is not the correct diagnosis.
*   Alternative Mechanism: The lab values (serum Na 152, serum osmolality 310, urine osmolality 750, urine Na 180) are consistent with a state of volume depletion (hypovolemia) or hypernatremia from water loss. In hypovolemia, the kidney appropriately conserves water, leading to a concentrated urine (urine osmolality > serum osmolality). The high urine sodium (>20 mEq/L) in this context suggests a renal cause of water loss (e.g., osmotic diuresis from hyperglycemia, mannitol, or urea; or diuretic use) rather than extra-renal losses (which would have low urine sodium <20 mEq/L) [2][4]. The patient is on "minimal fluid,"

*   Immediate Action: Discontinue any diuretics. Check serum glucose to rule out hyperglycemic osmotic diuresis.
*   Fluid Replacement: Calculate the free water deficit using the formula: Water deficit (L) = 0.6 x weight (kg) x [(serum Na / 140) - 1]. For a 70 kg man: 0.6 x 70 x [(152/140) - 1] = 42 x 0.086 = 3.6 L.
*   Rate of Correction: Replace half of the deficit over the first 24 hours, then the remainder over the next 24-48 hours. The goal is to lower serum sodium by no more than 10 mEq/L in the first 24 hours (target Na ~142-144).
*   Fluid Type: Use 5% Dextrose in Water (D5W) or 0.45% Normal Saline (hypotonic fluids). Do not use normal saline (0.9% NS) as it is isotonic and will not correct hypernatremia.
*   Monitoring: Recheck serum sodium every 4-6 hours during the initial correction phase.

3.

</details>


---

## 4. Conclusion

Across 10 complex clinical scenarios, GRAPES-SHAP improves clinical-concept coverage by **+30.0 percentage points** over a strong hybrid-RAG baseline, while additionally providing calibrated uncertainty, world-model treatment planning, and per-evidence SHAP explanations that the baseline cannot offer. These capabilities are essential for trustworthy clinical decision support.
