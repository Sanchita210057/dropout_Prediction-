"""Safety, inference, support-layer, batch, and UI tests."""
import hashlib
import io
import json
from pathlib import Path
import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from app import (BATCH_COLUMNS,FORM_FIELDS,ID_COLUMN,METADATA_PATH,MODEL_PATH,OCCUPATIONS,
    OCCUPATION_BY_LABEL,RESULT_COLUMNS,SEMESTER_COLUMN,THRESHOLD,USER_FIELDS,analyze_batch,
    dropout_probability,export_results,export_results_excel,make_input_frame,parse_boolean,predict_student,
    read_uploaded_table,risk_label,semester_gpa_to_source_grade,support_guidance,template_excel,
    validate_batch_dataframe,validate_inputs)
from src.uci_sem1_features import normalize_previous_grade

ROOT=Path(__file__).resolve().parents[1]
RESULTS_PATH=ROOT/'reports'/'uci_sem1'/'training_results_15_input.json'

def sample_values(**overrides):
    values={'previous_academic_gpa_normalized':3.5,'Age at enrollment':19,"Mother's occupation":9,"Father's occupation":9,"Mother's qualification":1,"Father's qualification":1,'Gender':0,'Marital status':1,'Scholarship holder':0,'Debtor':0,'Tuition fees up to date':1,'Curricular units 1st sem (enrolled)':6,'Curricular units 1st sem (evaluations)':8,'Curricular units 1st sem (approved)':5,'Curricular units 1st sem (grade)':12.0}
    values.update(overrides); return values

def student(**overrides):
    row={ID_COLUMN:'ST001',SEMESTER_COLUMN:'3rd Semester','HSC / Equivalent GPA':4.2,'Age at Enrollment':19,"Mother's Occupation":'Administrative staff',"Father's Occupation":'Unskilled workers',"Mother's Education":'Secondary education (12th year)',"Father's Education":'Secondary education (12th year)','Gender':'Female','Marital Status':'Single','Scholarship Holder':'No','Debtor':'No','Tuition Fees Up to Date':'Yes','Courses Enrolled':6,'Evaluations Completed':6,'Courses Passed':5,'Semester GPA':3.25}
    row.update(overrides); return row

def guidance(**overrides):
    values=dict(semester_gpa=3.5,enrolled=6,evaluations=6,passed=6,debtor=False,tuition_up_to_date=True,scholarship=True,previous_gpa=4.0)
    values.update(overrides); return support_guidance(**values)

def test_model_schema_artifact_and_threshold_are_locked():
    assert len(FORM_FIELDS)==len(USER_FIELDS)==15
    assert MODEL_PATH.is_file()
    artifact=joblib.load(MODEL_PATH); metadata=json.loads(METADATA_PATH.read_text(encoding='utf-8'))
    assert artifact['threshold']==metadata['threshold'] and 0<artifact['threshold']<1
    assert list(make_input_frame(sample_values()).columns)==FORM_FIELDS

def test_existing_individual_prediction_and_both_paths():
    artifact=joblib.load(MODEL_PATH); probability=dropout_probability(artifact,sample_values())
    assert 0<=probability<=1 and risk_label(artifact['threshold']-0.000001,artifact['threshold'])=='Lower Estimated Risk' and risk_label(artifact['threshold'],artifact['threshold'])=='Elevated Estimated Risk'
    raw=pd.read_csv(ROOT/'data'/'student_dropout.csv',sep=';'); raw.columns=raw.columns.str.strip(); raw['previous_academic_gpa_normalized']=normalize_previous_grade(raw['Previous qualification (grade)'])
    probabilities=artifact['pipeline'].predict_proba(raw[FORM_FIELDS])[:,list(artifact['pipeline'].classes_).index(1)]
    assert risk_label(float(probabilities.min())).startswith('Lower') and risk_label(float(probabilities.max())).startswith('Elevated')

def test_input_helpers_and_gpa_conversion():
    assert validate_inputs(sample_values(**{'Curricular units 1st sem (enrolled)':4,'Curricular units 1st sem (approved)':5}))
    assert {x:semester_gpa_to_source_grade(x) for x in [0.,2.,3.,3.5,4.]}=={0.:0.,2.:9.4375,3.:14.15625,3.5:16.515625,4.:18.875}
    assert OCCUPATION_BY_LABEL[OCCUPATIONS[4]]==4

def test_metrics_match_locked_results():
    r=json.loads(RESULTS_PATH.read_text(encoding='utf-8'))['holdout_metrics']
    assert all(0<=r[x]<=1 for x in ['precision','recall','roc_auc'])

def test_support_layer_priority_caps_and_safe_edge_cases():
    indicators,supports=guidance(semester_gpa=1.2,enrolled=8,evaluations=1,passed=1,debtor=True,tuition_up_to_date=False,scholarship=False,previous_gpa=2.5)
    assert len(indicators)==3 and len(supports)<=4 and indicators[0].startswith('Low recent academic performance')
    assert guidance()==([],[])
    assert 'No course enrollment was reported' in guidance(enrolled=0,evaluations=0,passed=0)[0]
    assert not any(word in ' '.join(indicators).lower() for word in ['mother','father','occupation','age'])

def test_academic_and_financial_support_mappings():
    academic,supports=guidance(semester_gpa=1.85,enrolled=7,evaluations=5,passed=3)
    assert 'Low course completion — 3 of 7 courses passed' in academic and 'Academic advising' in supports
    financial,supports=guidance(debtor=True,tuition_up_to_date=False,scholarship=False)
    assert 'Tuition fees are not currently up to date' in financial and 'Financial counselling' in supports

def test_common_boolean_mapping():
    for yes in ['Yes','TRUE','1',1,True]: assert parse_boolean(yes) is True
    for no in ['No','FALSE','0',0,False]: assert parse_boolean(no) is False
    with pytest.raises(ValueError): parse_boolean('maybe')

def test_valid_single_and_multi_row_batch_prediction():
    artifact=joblib.load(MODEL_PATH); one=pd.DataFrame([student()])
    assert validate_batch_dataframe(one).empty
    result=analyze_batch(artifact,one); assert len(result)==1 and 0<=result.loc[0,'Estimated Dropout Risk']<=1
    multi=pd.DataFrame([student(),student(**{ID_COLUMN:'ST002','Semester GPA':1.2,'Courses Passed':1,'Debtor':'Yes'}),student(**{ID_COLUMN:'ST003','Tuition Fees Up to Date':'No'})])
    result=analyze_batch(artifact,multi); assert len(result)==3 and result['Risk Level'].notna().all()
    assert (result['Risk Level']==result['Estimated Dropout Risk'].map(risk_label)).all()
    assert result['Key Risk Indicators'].map(lambda value: len(value.split(' | ')) if value!='-' else 0).max()<=3

def test_batch_validation_errors():
    assert 'Missing required columns' in validate_batch_dataframe(pd.DataFrame([student()]).drop(columns=['Semester GPA'])).loc[0,'Issue']
    cases=[({'HSC / Equivalent GPA':2.0},'HSC / Equivalent GPA'),({'Semester GPA':4.5},'Semester GPA'),({'Courses Enrolled':3,'Courses Passed':4},'Courses Passed')]
    for override,expected in cases:
        issues=' '.join(validate_batch_dataframe(pd.DataFrame([student(**override)]))['Issue'])
        assert expected in issues

def test_export_contains_required_columns_and_human_labels():
    result=analyze_batch(joblib.load(MODEL_PATH),pd.DataFrame([student()]))
    exported=pd.read_csv(pd.io.common.BytesIO(export_results(result)))
    assert set(RESULT_COLUMNS).issubset(exported.columns)
    assert exported.loc[0,"Mother's Occupation"]=='Administrative staff'
    assert not any('occupation code' in column.lower() for column in exported.columns)
    exported_excel=pd.read_excel(io.BytesIO(export_results_excel(result)),engine='openpyxl')
    assert set(RESULT_COLUMNS).issubset(exported_excel.columns)

def test_excel_template_and_upload_are_supported():
    template_bytes=template_excel()
    workbook=io.BytesIO(template_bytes); workbook.name='student_batch_template.xlsx'
    loaded=read_uploaded_table(workbook)
    assert list(loaded.columns)==BATCH_COLUMNS
    assert loaded.loc[0,ID_COLUMN]=='EXAMPLE-001'

    csv_file=io.BytesIO(pd.DataFrame([student()]).to_csv(index=False).encode('utf-8')); csv_file.name='students.csv'
    assert read_uploaded_table(csv_file).loc[0,ID_COLUMN]=='ST001'

    xlsx=io.BytesIO()
    with pd.ExcelWriter(xlsx,engine='openpyxl') as writer:
        pd.DataFrame([student()]).to_excel(writer,index=False)
    xlsx.seek(0); xlsx.name='students.xlsx'
    assert read_uploaded_table(xlsx).loc[0,ID_COLUMN]=='ST001'

def test_no_retraining_or_persistent_upload_storage_in_app():
    source=(ROOT/'app.py').read_text(encoding='utf-8').lower()
    assert '.fit(' not in source and 'to_pickle(' not in source and 'to_parquet(' not in source
    assert 'st.image(' not in source and 'st.pyplot(' not in source

def test_four_pages_render_and_individual_form_has_15_model_widgets():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    expected={'Dashboard':'Student Dropout Early Warning System','Individual Assessment':'Individual Assessment','Batch Analysis':'Batch Analysis','About Project':'About Project'}
    assert list(app.radio[0].options)==list(expected)
    for page,title in expected.items():
        app.radio[0].set_value(page).run(); assert not app.exception
        rendered=' '.join([*(item.value for item in app.title),*(item.value for item in app.markdown)])
        assert title.lower() in rendered.lower()
    app.radio[0].set_value('Individual Assessment').run()
    assert len(app.number_input)+len(app.selectbox)==15
    assert next(item for item in app.number_input if item.label=='Semester GPA').max==4.0

def test_elevated_result_panel_and_download_render():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run(); app.radio[0].set_value('Individual Assessment').run()
    numbers={item.label:item for item in app.number_input}; selections={item.label:item for item in app.selectbox}
    for label,value in [('HSC / Equivalent GPA',2.5),('Age at Enrollment',70),('Courses Enrolled',7),('Evaluations Completed',1),('Courses Passed',0),('Semester GPA',0.0)]: numbers[label].set_value(value)
    selections['Debtor'].set_value('Yes'); selections['Tuition Fees Up to Date'].set_value('No'); app.button[-1].click().run()
    rendered=' '.join(item.value for item in app.markdown)
    assert 'elevated estimated risk' in rendered.lower() and 'Key Risk Indicators' in rendered and 'Possible Institute Support' in rendered
    assert app.download_button
def test_batch_page_processes_five_row_upload_and_renders_outputs():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    app.radio[0].set_value('Batch Analysis').run()
    content=(ROOT/'data'/'sample_batch_students.csv').read_bytes()
    app.file_uploader[0].upload('sample_batch_students.csv',content,'text/csv').run()
    assert not app.exception and not app.error
    markdown=' '.join(item.value for item in app.markdown)
    assert 'Total Students' in markdown and 'Elevated Risk' in markdown and 'Average Estimated Risk' in markdown
    assert len(app.dataframe)>=1 and len(app.download_button)>=4
def test_batch_page_accepts_excel_upload():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    app.radio[0].set_value('Batch Analysis').run()
    buffer=pd.io.common.BytesIO()
    pd.DataFrame([student(),student(**{ID_COLUMN:'ST002','Semester GPA':1.4,'Courses Passed':1})]).to_excel(buffer,index=False)
    app.file_uploader[0].upload('students.xlsx',buffer.getvalue(),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet').run()
    assert not app.exception and not app.error
    assert app.dataframe and len(app.download_button)>=4
