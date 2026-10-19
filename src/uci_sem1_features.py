"""Derived features for the 15-input UCI Semester-1 model."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

NORMALIZED_GPA_FEATURE='previous_academic_gpa_normalized'
FINAL_FEATURES=[NORMALIZED_GPA_FEATURE,'Age at enrollment',"Mother's occupation","Father's occupation","Mother's qualification","Father's qualification",'Gender','Marital status','Scholarship holder','Debtor','Tuition fees up to date','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)','Curricular units 1st sem (grade)','Semester pass rate']
RAW_INPUT_FEATURES=[f for f in FINAL_FEATURES if f!='Semester pass rate']+['Curricular units 1st sem (approved)']

def normalize_previous_grade(values):
    return values.astype(float)/190.0*5.0

class SemesterFeatureBuilder(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None): return self
    def transform(self,X):
        frame=X.copy(); enrolled=frame['Curricular units 1st sem (enrolled)'].astype(float); approved=frame.pop('Curricular units 1st sem (approved)').astype(float)
        frame['Semester pass rate']=np.divide(approved,enrolled,out=np.zeros(len(frame),dtype=float),where=enrolled.to_numpy()!=0)
        return frame[FINAL_FEATURES]
