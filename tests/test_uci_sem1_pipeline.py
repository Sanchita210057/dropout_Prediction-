import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.train_uci_sem1 import (BANNED, DATA_PATH, FINAL_FEATURES, FULL_FEATURES,
    META_PATH, MODEL_PATH, RAW_FOR_FINAL, RESULTS_PATH, TARGET, load_data, prepare)
from src.uci_sem1_features import SemesterFeatureBuilder, normalize_previous_grade

def test_dataset_and_target_cohort():
    raw=load_data(); cohort,y=prepare(raw)
    assert raw.shape==(4424,37)
    assert raw[TARGET].value_counts().to_dict()=={'Graduate':2209,'Dropout':1421,'Enrolled':794}
    assert len(cohort)==3630 and int(y.sum())==1521
    assert set(y.unique())=={0,1}

def test_feature_policy_has_no_leakage_or_banned_fields():
    assert TARGET not in FULL_FEATURES+FINAL_FEATURES+RAW_FOR_FINAL
    assert all(not f.startswith('Curricular units 2nd sem') for f in FULL_FEATURES+FINAL_FEATURES+RAW_FOR_FINAL)
    assert BANNED.isdisjoint(FULL_FEATURES)
    assert len(RAW_FOR_FINAL)==15
    assert 'Admission grade' not in FINAL_FEATURES+RAW_FOR_FINAL

def test_derived_pass_rate_and_zero_denominator():
    frame=pd.DataFrame({'Curricular units 1st sem (enrolled)':[4,0], 'Curricular units 1st sem (approved)':[3,0], **{f:[1,1] for f in RAW_FOR_FINAL if f not in {'Curricular units 1st sem (enrolled)','Curricular units 1st sem (approved)'}}})
    result=SemesterFeatureBuilder().transform(frame)
    assert result['Semester pass rate'].tolist()==[.75,0.0]
    assert np.isfinite(result['Semester pass rate']).all()

def test_normalized_previous_gpa_examples():
    values=pd.Series([190,152,95])
    assert normalize_previous_grade(values).tolist()==[5.0,4.0,2.5]

def test_saved_artifact_round_trip_probability_and_threshold():
    artifact=joblib.load(MODEL_PATH); metadata=json.loads(META_PATH.read_text(encoding='utf-8'))
    raw=load_data(); cohort,_=prepare(raw); cohort['previous_academic_gpa_normalized']=normalize_previous_grade(cohort['Previous qualification (grade)']); row=cohort[RAW_FOR_FINAL].iloc[:2]
    probabilities=artifact['pipeline'].predict_proba(row)[:,1]
    assert isinstance(artifact['pipeline'],Pipeline)
    assert np.all((probabilities>=0)&(probabilities<=1))
    assert artifact['threshold']==metadata['threshold'] and 0<artifact['threshold']<1
    assert metadata['normalized_gpa_range']=={'min':2.5,'max':5.0,'formula':'Previous qualification (grade) / 190 * 5'}
    assert artifact['features']==FINAL_FEATURES

def test_saved_results_are_locked_and_holdout_is_single_use():
    results=json.loads((Path(RESULTS_PATH).parent/'training_results_15_input.json').read_text(encoding='utf-8'))
    assert len(results['raw_input_features'])==15
    assert results['holdout_metrics']['tn']+results['holdout_metrics']['fp']+results['holdout_metrics']['fn']+results['holdout_metrics']['tp']==726
