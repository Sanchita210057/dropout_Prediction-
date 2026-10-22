"""Local Student Dropout Early Warning System."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import io
import joblib
import pandas as pd
import streamlit as st
from src.bangladesh_collection import (
    EXPORT_COLUMNS as BD_EXPORT_COLUMNS,
    FIELD_TITLES as BD_FIELD_TITLES,
    CATEGORICAL_CHOICES as BD_CATEGORICAL_CHOICES,
    PREDICTOR_FIELDS as BD_PREDICTOR_FIELDS,
    STATUS_FIELD as BD_STATUS_FIELD,
    STATUS_TITLE as BD_STATUS_TITLE,
    ACADEMIC_STATUS_OPTIONS as BD_ACADEMIC_STATUS_OPTIONS,
    academic_status_target,
    clean_google_forms_responses,
    questionnaire_template_frame,
    validate_bangladesh_responses,
)
from src.combined_ensemble_v2 import (
    UCI_OCCUPATIONS,
    parent_occupation_to_uci,
)

ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/'models'/'combined_uci_bangladesh_18_v2'/'weighted_ensemble_dropout_status_18.joblib'
METADATA_PATH=ROOT/'models'/'combined_uci_bangladesh_18_v2'/'model_metadata.json'
THRESHOLD=None
FORM_FIELDS=list(BD_PREDICTOR_FIELDS)
USER_FIELDS=[BD_FIELD_TITLES[field] for field in FORM_FIELDS]
USER_FIELD_TO_KEY=dict(zip(USER_FIELDS,FORM_FIELDS))
ASSESSMENT_CATEGORICAL_CHOICES=dict(BD_CATEGORICAL_CHOICES)
ID_COLUMN='Anonymous Follow-Up ID'
SEMESTER_COLUMN='Current Semester / Academic Year'
BATCH_COLUMNS=[ID_COLUMN,SEMESTER_COLUMN,*USER_FIELDS]
RESULT_COLUMNS=[ID_COLUMN,SEMESTER_COLUMN,'Estimated Current Dropout-Status Probability','UCI Academic Foundation Signal (research estimate)','Bangladesh Local Model Signal (research estimate)','Risk Level','Main Indicator','Key Risk Indicators','Suggested Institute Support']
APP_PAGES=['Dashboard','Individual Assessment','Batch Analysis','About Project']
OCCUPATIONS=UCI_OCCUPATIONS
OCCUPATION_BY_LABEL={label:code for code,label in OCCUPATIONS.items()}
EDUCATION_LEVELS={1:'Secondary education (12th year)',2:'Bachelor degree',3:'Higher education degree',4:'Master degree',5:'Doctorate',6:'Higher education (some)',9:'Incomplete secondary (12th year)',10:'Incomplete 11th year',11:'7th year (old)',12:'11th year (other)',14:'10th year',18:'General commerce course',19:'Basic education 3rd cycle',22:'Technical-professional course',26:'7th year',27:'2nd cycle general high school',29:'Incomplete 9th year',30:'8th year',34:'Unknown',35:'Cannot read or write',36:'Reads without 4th year',37:'Basic education 1st cycle',38:'Basic education 2nd cycle',39:'Technological specialization',40:'Higher degree (1st cycle)',41:'Specialized higher studies',42:'Professional higher technical',43:'Master (2nd cycle)',44:'Doctorate (3rd cycle)'}
EDUCATION_BY_LABEL={label:code for code,label in EDUCATION_LEVELS.items()}
GENDERS={'Female':0,'Male':1}
MARITAL_STATUSES={'Single':1,'Married':2,'Widower':3,'Divorced':4,'De facto union':5,'Legally separated':6}
MARITAL_BY_CODE={v:k for k,v in MARITAL_STATUSES.items()}
def marital_code(value:Any)->int:
    if value in MARITAL_STATUSES: return MARITAL_STATUSES[value]
    try: code=int(float(value))
    except (TypeError,ValueError): raise ValueError('must be a supported marital-status label or code')
    if code not in MARITAL_BY_CODE: raise ValueError('must be a supported marital-status label or code')
    return code
def education_code(value:Any)->int:
    if value in EDUCATION_BY_LABEL: return EDUCATION_BY_LABEL[value]
    try: code=int(float(value))
    except (TypeError,ValueError): raise ValueError('is not a supported education label')
    if code not in EDUCATION_LEVELS: raise ValueError('is not a supported education label')
    return code
def gender_code(value:Any)->int:
    if value in GENDERS: return GENDERS[value]
    try: code=int(float(value))
    except (TypeError,ValueError): raise ValueError('must be Female/Male or 0/1')
    if code not in (0,1): raise ValueError('must be Female/Male or 0/1')
    return code


@st.cache_resource
def load_model_artifact()->dict[str,Any]:
    if not MODEL_PATH.is_file():
        raise RuntimeError(f'Combined UCI + Bangladesh 18-input model artifact is missing: {MODEL_PATH}. Run: python -m src.train_combined_ensemble_18')
    artifact=joblib.load(MODEL_PATH)
    pipeline_fields=list(getattr(artifact.get('pipeline'),'feature_names_in_',[]))
    if (artifact.get('raw_input_features')!=FORM_FIELDS or pipeline_fields!=FORM_FIELDS
            or artifact.get('input_feature_count')!=18 or artifact.get('target_is_predictor') is not False
            or artifact.get('target_field')!=BD_STATUS_FIELD
            or artifact.get('architecture','').startswith('UCI 10-feature') is False
            or getattr(artifact.get('pipeline'),'blend_weight_uci',0)<=0):
        raise RuntimeError('The active combined model does not match the exact 18-input schema or lacks a UCI contribution. Run: python -m src.train_combined_ensemble_18')
    return artifact

def make_input_frame(values:dict[str,Any])->pd.DataFrame:
    if set(values)!=set(FORM_FIELDS): raise ValueError('Answers must contain exactly the 18 model input fields.')
    normalized=clean_google_forms_responses(pd.DataFrame([{field:values[field] for field in FORM_FIELDS}]))
    frame=normalized.loc[:,FORM_FIELDS].copy()
    for field in FORM_FIELDS:
        if field in BD_CATEGORICAL_CHOICES:
            frame[field]=frame[field].astype(object).where(frame[field].notna(),float('nan'))
        else:
            frame[field]=pd.to_numeric(frame[field],errors='coerce').astype(float)
    return frame

def validate_inputs(values:dict[str,Any])->list[str]:
    issues=[]
    if set(values)!=set(FORM_FIELDS): return ['Answers must contain exactly the 18 model input fields.']
    if values['courses_passed']>values['courses_enrolled']: issues.append('Courses passed cannot be greater than courses enrolled.')
    for field,low,high in [('hsc_equivalent_gpa',0,5),('age_at_enrollment',15,100),('semester_gpa_native',0,5),('last_semester_attendance_percent',0,100)]:
        try:
            value=float(values[field])
            if value<low or value>high: issues.append(f'{BD_FIELD_TITLES[field]} must be between {low:g} and {high:g}.')
        except (TypeError,ValueError): issues.append(f'{BD_FIELD_TITLES[field]} must be numeric.')
    for field in ('courses_enrolled','evaluations_completed','courses_passed'):
        try:
            if int(values[field])!=float(values[field]) or float(values[field])<0: issues.append(f'{BD_FIELD_TITLES[field]} must be a non-negative whole number.')
        except (TypeError,ValueError): issues.append(f'{BD_FIELD_TITLES[field]} must be numeric.')
    for field,choices in ASSESSMENT_CATEGORICAL_CHOICES.items():
        if field in values and str(values[field]) not in choices: issues.append(f'{BD_FIELD_TITLES[field]} has an unsupported value.')
    return issues

def dropout_probability(artifact:dict[str,Any],values:dict[str,Any])->float:
    pipeline=artifact['pipeline']; index=list(pipeline.classes_).index(1)
    if artifact.get('raw_input_features')!=FORM_FIELDS or artifact.get('target_is_predictor') is not False:
        raise ValueError('The saved model does not support the exact combined 18-input schema.')
    return float(pipeline.predict_proba(make_input_frame(values))[0,index])

def risk_label(probability:float,threshold:float|None=None)->str:
    if threshold is None: threshold=load_model_artifact()['threshold']
    return 'Elevated Estimated Risk' if probability>=threshold else 'Lower Estimated Risk'

def support_guidance(*,semester_gpa:float,enrolled:int,evaluations:int,passed:int,debtor:bool,tuition_up_to_date:bool,scholarship:bool,previous_gpa:float)->tuple[list[str],list[str]]:
    pass_rate=passed/enrolled if enrolled else 0.0; evaluation_rate=evaluations/enrolled if enrolled else 0.0; candidates=[]
    if semester_gpa<2.0: candidates.append((10,f'Low recent academic performance — Semester GPA: {semester_gpa:.2f} on the reported scale',['Academic advising','Tutoring / study support']))
    if not enrolled: candidates.append((9,'No course enrollment was reported',['Study-plan review']))
    elif pass_rate<0.5: candidates.append((9,f'Low course completion — {passed} of {enrolled} courses passed',['Course-specific academic support','Study-plan review']))
    elif pass_rate<0.7: candidates.append((5,f'Moderate course completion — {passed} of {enrolled} courses passed',['Study-plan review']))
    if not tuition_up_to_date: candidates.append((8,'Tuition fees are not currently up to date',['Financial-aid review','Installment/payment support if available']))
    if debtor: candidates.append((8,'Outstanding debtor status was reported',['Financial counselling','Review possible payment arrangements']))
    if enrolled and evaluation_rate<0.75: candidates.append((7,f'Low evaluation completion — {evaluations} evaluations for {enrolled} enrolled courses',['Student engagement follow-up','Check attendance / participation difficulties']))
    if not scholarship and (debtor or not tuition_up_to_date): candidates.append((4,'No scholarship was reported alongside a financial warning',['Review scholarship or student-aid eligibility']))
    if previous_gpa<3.0: candidates.append((3,f'Weaker previous academic background — HSC / Equivalent GPA: {previous_gpa:.2f} / 5.00',['Foundation/remedial academic support if needed']))
    selected=sorted(candidates,key=lambda item:item[0],reverse=True)[:3]; supports=[]
    for _,_,recommendations in selected:
        if recommendations[0] not in supports: supports.append(recommendations[0])
    for _,_,recommendations in selected:
        for recommendation in recommendations[1:]:
            if recommendation not in supports and len(supports)<4: supports.append(recommendation)
    return [item[1] for item in selected],supports

def parse_boolean(value:Any)->bool:
    if isinstance(value,bool): return value
    if isinstance(value,(int,float)) and not pd.isna(value) and float(value) in (0,1): return bool(value)
    normalized=str(value).strip().lower()
    if normalized in {'yes','true','1'}: return True
    if normalized in {'no','false','0'}: return False
    raise ValueError('must be Yes/No, TRUE/FALSE, or 1/0')

def occupation_code(value:Any)->int:
    mapped=parent_occupation_to_uci(value)
    if mapped is not None: return mapped
    if value in OCCUPATION_BY_LABEL: return OCCUPATION_BY_LABEL[value]
    try: code=int(float(value))
    except (TypeError,ValueError): raise ValueError('is not a supported occupation label')
    if code not in OCCUPATIONS: raise ValueError('is not a supported occupation label')
    return code

def prepare_student(record:dict[str,Any])->dict[str,Any]:
    """Map visible batch/form titles to the same 18 Bangladesh model fields."""
    if set(USER_FIELDS).issubset(record):
        values={USER_FIELD_TO_KEY[title]:record[title] for title in USER_FIELDS}
    elif set(FORM_FIELDS).issubset(record):
        values={field:record[field] for field in FORM_FIELDS}
    else:
        missing=[title for title in USER_FIELDS if title not in record]
        raise ValueError('Missing 18-input assessment fields: '+', '.join(missing))
    for field in ('hsc_equivalent_gpa','age_at_enrollment','semester_gpa_native','last_semester_attendance_percent','courses_enrolled','evaluations_completed','courses_passed'):
        if pd.isna(values[field]) or str(values[field]).strip()=='': values[field]=float('nan')
        elif field in {'age_at_enrollment','courses_enrolled','evaluations_completed','courses_passed'}: values[field]=int(float(values[field]))
        else: values[field]=float(values[field])
    return values

def predict_student(artifact:dict[str,Any],record:dict[str,Any])->dict[str,Any]:
    values=prepare_student(record); probability=dropout_probability(artifact,values); label=risk_label(probability,artifact['threshold'])
    local_signal,uci_signal=artifact['pipeline'].component_probabilities(make_input_frame(values))
    indicators,supports=support_guidance(semester_gpa=float(values['semester_gpa_native']),enrolled=int(values['courses_enrolled']),evaluations=int(values['evaluations_completed']),passed=int(values['courses_passed']),debtor=str(values['debtor'])=='Yes',tuition_up_to_date=str(values['tuition_fees_up_to_date'])=='Yes',scholarship=str(values['scholarship_holder'])=='Yes',previous_gpa=float(values['hsc_equivalent_gpa']))
    attendance=values['last_semester_attendance_percent']
    if attendance<75: indicators.append(f'Low reported last-semester attendance — {attendance:.1f}%'); supports.insert(0,'Attendance follow-up and student check-in')
    if str(values['family_financial_condition']).startswith(('Poor','Below Average')): indicators.append('Reported household income is in a lower provisional survey bracket'); supports.append('Financial-aid review')
    if str(values['student_employment']) in {'Full-time','Part-time'}: indicators.append(f'Student reported {values["student_employment"].lower()} employment'); supports.append('Review study/work schedule and flexibility')
    return {'probability':probability,'risk_level':label,'threshold':artifact['threshold'],'component_scores':{'uci_academic':float(uci_signal[0]),'bangladesh_local':float(local_signal[0])},'indicators':indicators[:3],'supports':list(dict.fromkeys(supports))[:4]}

def validate_batch_dataframe(frame:pd.DataFrame)->pd.DataFrame:
    missing=[column for column in USER_FIELDS if column not in frame.columns]
    if missing: return pd.DataFrame([{'Row':'File','Issue':'Missing required columns: '+', '.join(missing)}])
    issues=[]
    ranges={'HSC / Equivalent GPA':(0,5,False),'Age at enrollment':(15,100,True),'Courses enrolled in the relevant semester':(0,500,True),'Evaluations completed in the relevant semester':(0,1000,True),'Courses passed in the relevant semester':(0,500,True),'Semester GPA (native university scale)':(0,5,False),'Last-semester attendance percentage':(0,100,False)}
    for position,(_,row) in enumerate(frame.iterrows(),start=2):
        canonical={USER_FIELD_TO_KEY[col]:row[col] for col in USER_FIELDS}
        def issue(message:str): issues.append({'Row':position,'Issue':message})
        for title,(minimum,maximum,integer) in ranges.items():
            value=row[title]
            if pd.isna(value) or str(value).strip()=='': continue
            try: number=float(value)
            except (TypeError,ValueError): issue(f'{title} must be numeric.'); continue
            if number<minimum or number>maximum or (integer and not number.is_integer()):
                kind='whole number' if integer else 'number'
                issue(f'{title} must be a {kind} in range {minimum:g}–{maximum:g}.')
        for field,choices in ASSESSMENT_CATEGORICAL_CHOICES.items():
            if field not in FORM_FIELDS: continue
            title=BD_FIELD_TITLES[field]; value=row[title]
            if pd.isna(value) or str(value).strip()=='': continue
            normalized=str(value).strip()
            if field=='marital_status_bangladesh' and normalized=='Single': normalized='Single / Never married'
            if normalized not in choices: issue(f'{title} has an unsupported value.')
        try:
            passed=canonical['courses_passed']; enrolled=canonical['courses_enrolled']
            if not pd.isna(passed) and not pd.isna(enrolled) and float(passed)>float(enrolled): issue('Courses passed cannot be greater than courses enrolled.')
        except (TypeError,ValueError): pass
    return pd.DataFrame(issues,columns=['Row','Issue'])

def indicator_group(indicator:str)->str:
    return 'Financial' if any(word in indicator.lower() for word in ['tuition','debtor','scholarship']) else 'Academic'

def metadata_text(value:Any,fallback:str='')->str:
    return fallback if value is None or pd.isna(value) or not str(value).strip() else str(value).strip()

def analyze_batch(artifact:dict[str,Any],frame:pd.DataFrame)->pd.DataFrame:
    errors=validate_batch_dataframe(frame)
    if not errors.empty: raise ValueError(errors.to_json(orient='records'))
    rows=[]
    for position,(_,source) in enumerate(frame.iterrows(),start=1):
        record=source.to_dict(); result=predict_student(artifact,record); indicators=result['indicators'] if result['risk_level'].startswith('Elevated') else []
        rows.append({**record,ID_COLUMN:metadata_text(record.get(ID_COLUMN),f'Student {position}'),SEMESTER_COLUMN:metadata_text(record.get(SEMESTER_COLUMN)), 'Estimated Current Dropout-Status Probability':result['probability'],'UCI Academic Foundation Signal (research estimate)':result['component_scores']['uci_academic'],'Bangladesh Local Model Signal (research estimate)':result['component_scores']['bangladesh_local'],'Risk Level':result['risk_level'],'Main Indicator':indicators[0] if indicators else '-','Key Risk Indicators':' | '.join(indicators) if indicators else '-','Suggested Institute Support':' | '.join(result['supports']) if indicators else '-','Warning Type':indicator_group(indicators[0]) if indicators else '-'})
    return pd.DataFrame(rows)

def export_results(frame:pd.DataFrame)->bytes:
    columns=[*BATCH_COLUMNS,*RESULT_COLUMNS[2:]]
    return frame[[column for column in columns if column in frame.columns]].to_csv(index=False).encode('utf-8')

def template_frame()->pd.DataFrame:
    example={ID_COLUMN:'EXAMPLE-001',SEMESTER_COLUMN:'Relevant semester'}
    for field in FORM_FIELDS:
        title=BD_FIELD_TITLES[field]
        choices=ASSESSMENT_CATEGORICAL_CHOICES.get(field)
        if choices: example[title]=choices[0]
        elif field=='hsc_equivalent_gpa': example[title]=4.20
        elif field=='age_at_enrollment': example[title]=19
        elif field in {'courses_enrolled','evaluations_completed','courses_passed'}: example[title]=5
        elif field=='semester_gpa_native': example[title]=3.25
        elif field=='last_semester_attendance_percent': example[title]=85.0
    return pd.DataFrame([example],columns=BATCH_COLUMNS)

def template_csv()->bytes:
    return template_frame().to_csv(index=False).encode('utf-8')

def metric_cards(items:list[tuple[str,str,str]])->None:
    for column,(value,label,icon) in zip(st.columns(len(items)),items):
        column.markdown(f'<div class="metric-card"><div class="metric-icon">{icon}</div><strong>{value}</strong><span>{label}</span></div>',unsafe_allow_html=True)

def section_heading(title:str,eyebrow:str='')->None:
    eyebrow_html=f'<span>{eyebrow}</span>' if eyebrow else ''
    st.markdown(f'<div class="section-heading">{eyebrow_html}<h3>{title}</h3></div>',unsafe_allow_html=True)

def info_card(title:str,body:str)->None:
    st.markdown(f'<div class="info-card"><h3>{title}</h3><div>{body}</div></div>',unsafe_allow_html=True)

def assessment_pdf(result:dict[str,Any],student_id:str)->bytes:
    """Format the already-calculated assessment as a compact vector PDF without rerunning prediction."""
    commands=[]
    def text(x:float,y:float,value:Any,size:int=10,color:tuple[float,float,float]=(.18,.25,.34),font:str='F1')->None:
        # Built-in PDF fonts use WinAnsi; replace unsupported glyphs safely for portability.
        content=str(value).encode('cp1252','replace').decode('cp1252')
        escaped=content.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
        commands.append(f'BT /{font} {size} Tf {color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg {x:.1f} {y:.1f} Td ({escaped}) Tj ET')
    def rect(x:float,y:float,w:float,h:float,color:tuple[float,float,float],stroke:tuple[float,float,float]|None=None)->None:
        commands.append(f'{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg')
        if stroke: commands.append(f'{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG 0.8 w')
        commands.append(f'{x:.1f} {y:.1f} {w:.1f} {h:.1f} re '+('B' if stroke else 'f'))
    def wrap(value:Any,limit:int=82)->list[str]:
        words=str(value).split(); lines=[]; line=''
        for word in words:
            candidate=f'{line} {word}'.strip()
            if len(candidate)>limit and line: lines.append(line); line=word
            else: line=candidate
        if line: lines.append(line)
        return lines or ['']

    rect(0,632,612,160,(.035,.095,.16))
    rect(42,632,5,160,(.10,.76,.88))
    text(62,742,'STUDENT RISK ASSESSMENT',9,(.36,.84,.94),'F2')
    text(62,708,'Student Dropout Early Warning System',19,(1,1,1),'F2')
    rect(42,536,528,74,(.93,.965,.98),(.75,.83,.88))
    text(58,583,'STUDENT REFERENCE',8,(.30,.42,.53),'F2')
    for i,line in enumerate(wrap(student_id,27)[:2]): text(58,562-i*13,line,10)
    text(246,583,'ESTIMATED RISK',8,(.30,.42,.53),'F2')
    text(246,552,f'{float(result["probability"]):.1%}',22,(.05,.30,.44),'F2')
    text(402,583,'CLASSIFICATION',8,(.30,.42,.53),'F2')
    for i,line in enumerate(wrap(result['risk_level'],24)[:2]): text(402,562-i*13,line,10,(.13,.39,.45),'F2')
    y=490
    for heading,key,fallback in [('KEY RISK INDICATORS','indicators','No specific indicators were reported.'),
                                 ('RECOMMENDED INSTITUTE SUPPORT','supports','Continue normal academic monitoring.')]:
        text(46,y,heading,11,(.04,.45,.58),'F2'); y-=12
        commands.append(f'0.12 0.79 0.87 RG 1.4 w 46 {y:.1f} m 566 {y:.1f} l S'); y-=19
        items=result.get(key) or [fallback]
        for item in items:
            lines=wrap(item)
            for index,line in enumerate(lines):
                text(52,y,'•' if index==0 else ' ',10,(.07,.62,.70),'F2')
                text(68,y,line,10); y-=15
            y-=4
        y-=12
    text(46,38,'Assessment formatted from the result displayed in the application.',8,(.45,.52,.60))
    stream=('\n'.join(commands)+'\n').encode('cp1252','replace')
    objects=[b'<< /Type /Catalog /Pages 2 0 R >>',
             b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
             b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>',
             b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>',
             b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>',
             b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'endstream']
    pdf=bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'); offsets=[0]
    for number,obj in enumerate(objects,start=1):
        offsets.append(len(pdf)); pdf.extend(f'{number} 0 obj\n'.encode()); pdf.extend(obj); pdf.extend(b'\nendobj\n')
    xref=len(pdf); pdf.extend(f'xref\n0 {len(objects)+1}\n'.encode()); pdf.extend(b'0000000000 65535 f \n')
    for offset in offsets[1:]: pdf.extend(f'{offset:010d} 00000 n \n'.encode())
    pdf.extend(f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF'.encode())
    return bytes(pdf)

def render_performance_chart(metrics:dict[str,Any])->None:
    specs=[('Accuracy','accuracy'),('Precision','precision'),('Recall','recall'),('F1-score','f1')]
    rows=[]
    for label,key in specs:
        value=float(metrics[key]); percent=f'{value:.1%}'
        rows.append(f'<div class="performance-row"><span>{label}</span><div class="performance-track"><i style="width:{max(0,min(value,1))*100:.1f}%"></i></div><b>{percent}</b></div>')
    supporting=''.join(f'<div class="support-metric"><span>{label}</span><strong>{float(metrics[key]):.1%}</strong></div>' for label,key in [('ROC-AUC','roc_auc'),('PR-AUC','pr_auc_average_precision')])
    st.markdown('<div class="performance-chart">'+''.join(rows)+'</div><div class="support-metrics">'+supporting+'</div>',unsafe_allow_html=True)

def render_confusion_matrix(cm:dict[str,int])->None:
    tn,fp,fn,tp=(int(cm[key]) for key in ('tn','fp','fn','tp'))
    correct=tn+tp; total=tn+fp+fn+tp; dropout_identified=tp; dropout_total=fn+tp
    st.markdown(f'''<div class="matrix-wrap"><table class="confusion-matrix" aria-label="Five-fold cross-validation confusion matrix">
      <thead><tr><th></th><th>Predicted: Lower risk</th><th>Predicted: Elevated risk</th></tr></thead>
      <tbody><tr><th>Actual: Non-dropout</th><td class="matrix-cell correct"><b>{tn}</b><small>TN · correct</small></td><td class="matrix-cell mistake"><b>{fp}</b><small>FP · incorrect alert</small></td></tr>
      <tr><th>Actual: Dropout</th><td class="matrix-cell mistake"><b>{fn}</b><small>FN · missed case</small></td><td class="matrix-cell correct"><b>{tp}</b><small>TP · identified</small></td></tr></tbody></table></div>
      <div class="matrix-summary"><div><strong>{correct} <small>/ {total}</small></strong><span>Correctly classified</span></div><div><strong>{dropout_identified} <small>/ {dropout_total}</small></strong><span>Dropout cases identified</span></div></div>
      <p class="matrix-legend"><b>Five-fold cross-validation</b> · TN correct lower-risk result · FP incorrect elevated-risk alert · FN missed dropout case · TP correctly identified dropout case</p>''',unsafe_allow_html=True)

def render_model_architecture()->None:
    st.markdown('''<div class="model-architecture"><div class="model-branches">
      <div class="model-branch"><span>UCI Dataset · 3,630 records</span><b>Logistic Regression academic model</b><small>10 aligned features</small></div>
      <div class="model-branch"><span>Bangladesh Dataset · 251 records</span><b>Logistic Regression local model</b><small>18 questionnaire features</small></div>
      </div><div class="architecture-merge">Both component predictions ↓</div>
      <div class="architecture-steps"><span>Weighted combination</span><b>→</b><span>Logistic Regression calibration</span><b>→</b><strong>Final assessment</strong></div></div>''',unsafe_allow_html=True)

def dataframe_to_excel(frame:pd.DataFrame,sheet_name:str)->bytes:
    buffer=io.BytesIO()
    with pd.ExcelWriter(buffer,engine='openpyxl') as writer:
        frame.to_excel(writer,index=False,sheet_name=sheet_name)
    return buffer.getvalue()

def bangladesh_research_excel(frame:pd.DataFrame)->bytes:
    """Export only the 18 Bangladesh questionnaire predictors."""
    return dataframe_to_excel(frame,'Bangladesh Survey v2')

def template_excel()->bytes:
    return dataframe_to_excel(template_frame(),'Student Template')

def export_results_excel(frame:pd.DataFrame)->bytes:
    columns=[*BATCH_COLUMNS,*RESULT_COLUMNS[2:]]
    selected=frame[[column for column in columns if column in frame.columns]]
    return dataframe_to_excel(selected,'Analysis Results')

def read_uploaded_table(uploaded:Any)->pd.DataFrame:
    name=str(getattr(uploaded,'name','')).lower()
    if hasattr(uploaded,'seek'): uploaded.seek(0)
    if name.endswith('.csv'): return pd.read_csv(uploaded)
    if name.endswith('.xlsx'): return pd.read_excel(uploaded,engine='openpyxl')
    raise ValueError('Unsupported file type. Please upload a CSV or Excel (.xlsx) file.')

def navigate(page:str)->None: st.session_state['page']=page

def dashboard(artifact:dict[str,Any])->None:
    metrics=artifact.get('active_model_metrics',{})
    accuracy=f'{metrics["accuracy"]:.1%}' if 'accuracy' in metrics else '—'
    recall=f'{metrics["recall"]:.1%}' if 'recall' in metrics else '—'
    st.markdown('''<div class="hero"><h1>Student Dropout Early Warning System</h1><h2>AI-powered student risk assessment using academic and local student factors.</h2></div>''',unsafe_allow_html=True)
    metric_cards([('Logistic Regression','Ensemble','◈'),('18','Student Factors','◆'),(accuracy,'Accuracy','◉'),(recall,'Recall','↗')])
    st.markdown('<div class="process-strip"><span>Student Data</span><b>→</b><span>Risk Analysis</span><b>→</b><span>Warning Indicators</span><b>→</b><span>Support Action</span></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2,gap='large')
    c1.markdown('<div class="action-copy"><b>Individual screening</b><span>Review one student with an immediate support summary.</span></div>',unsafe_allow_html=True)
    c1.button('Assess One Student',type='primary',use_container_width=True,key='dashboard_assess_one',on_click=navigate,args=('Individual Assessment',))
    c2.markdown('<div class="action-copy"><b>Department overview</b><span>Upload a cohort file and identify support priorities.</span></div>',unsafe_allow_html=True)
    c2.button('Analyze Student Batch',type='primary',use_container_width=True,key='dashboard_analyze_batch',on_click=navigate,args=('Batch Analysis',))

def render_result(result:dict[str,Any],student_id:str='Individual Assessment')->None:
    elevated=result['risk_level'].startswith('Elevated'); status='elevated' if elevated else 'lower'; width=max(2,min(100,result['probability']*100))
    st.markdown(f'''<div class="result-panel {status}"><div class="result-top"><div><span class="result-label">ESTIMATED REPORTED CURRENT DROPOUT-STATUS PROBABILITY</span><strong>{result["probability"]:.1%}</strong></div><div class="status-badge">{result["risk_level"]}</div></div><div class="risk-track"><div style="width:{width:.1f}%"></div></div><div class="scale"><span>Lower estimate</span><span>Threshold {result["threshold"]:.1%}</span><span>Higher estimate</span></div></div>''',unsafe_allow_html=True)
    components=result.get('component_scores')
    if components:
        with st.expander('Model Details',expanded=False):
            left,right=st.columns(2,gap='medium')
            left.metric('Academic Model Score',f'{components["uci_academic"]:.1%}')
            right.metric('Local Student Model Score',f'{components["bangladesh_local"]:.1%}')
    section_heading('Key Risk Indicators','INTERPRETATION')
    indicators=result['indicators'] or ['No specific risk indicators identified.']
    st.markdown('<div class="chip-grid">'+''.join(f'<div class="indicator-chip"><i>!</i><span>{item}</span></div>' for item in indicators)+'</div>',unsafe_allow_html=True)
    section_heading('Possible Institute Support','RECOMMENDATIONS')
    supports=result['supports'] or ['Continue normal academic monitoring.']
    st.markdown('<div class="support-grid">'+''.join(f'<div class="support-card"><i>→</i><span>{item}</span></div>' for item in supports)+'</div>',unsafe_allow_html=True)
    st.download_button('↓ Download Assessment PDF',assessment_pdf(result,student_id),file_name='student_assessment.pdf',mime='application/pdf')

def individual_assessment(artifact:dict[str,Any])->None:
    st.markdown('<div class="page-header"><span>INDIVIDUAL SCREENING</span><h1>Individual Assessment</h1><p>Enter student information to generate a dropout risk assessment.</p></div>',unsafe_allow_html=True)
    with st.form('assessment_form'):
        student_id=st.text_input('Anonymous Student ID (optional)',placeholder='e.g. STUDENT-001')
        answers={}
        section_heading('Student Background','01')
        for field in ('hsc_equivalent_gpa','age_at_enrollment','gender_bangladesh','marital_status_bangladesh'):
            choices=BD_CATEGORICAL_CHOICES.get(field)
            if choices:
                default='Unknown / Prefer not to say' if 'Unknown / Prefer not to say' in choices else choices[0]
                answers[field]=st.selectbox(BD_FIELD_TITLES[field],choices,index=choices.index(default),key='individual_'+field)
            elif field=='hsc_equivalent_gpa': answers[field]=st.number_input(BD_FIELD_TITLES[field],0.0,5.0,3.5,step=0.01)
            else: answers[field]=st.number_input(BD_FIELD_TITLES[field],15,100,19)
        section_heading('Family Background','02')
        for field in ('mother_occupation_bangladesh','father_occupation_bangladesh','mother_education_bangladesh','father_education_bangladesh'):
            choices=ASSESSMENT_CATEGORICAL_CHOICES[field]; default='Unknown / Prefer not to say'
            help_text=('Choose a Bangladesh questionnaire category. The UCI component uses ten harmonized academic/demographic features and does not learn parental occupation categories. The local component preserves Bangladesh categories; father Housewife was absent from its training vocabulary.' if 'occupation' in field else None)
            answers[field]=st.selectbox(BD_FIELD_TITLES[field],choices,index=choices.index(default),key='individual_'+field,help=help_text)
        section_heading('Financial Status','03')
        for field in ('scholarship_holder','debtor','tuition_fees_up_to_date','family_financial_condition','student_employment'):
            choices=BD_CATEGORICAL_CHOICES[field]
            default='Unknown / Prefer not to say' if 'Unknown / Prefer not to say' in choices else 'Undisclosed'
            answers[field]=st.selectbox(BD_FIELD_TITLES[field],choices,index=choices.index(default),key='individual_'+field)
        section_heading('Recent Semester Performance','04')
        for field in ('courses_enrolled','evaluations_completed','courses_passed','semester_gpa_native','last_semester_attendance_percent'):
            title=BD_FIELD_TITLES[field]
            if field in {'courses_enrolled','evaluations_completed','courses_passed'}: answers[field]=st.number_input(title,0,1000,5 if field!='evaluations_completed' else 5,key='individual_'+field)
            elif field=='semester_gpa_native': answers[field]=st.number_input(title,0.0,5.0,3.0,step=0.01,key='individual_'+field)
            else: answers[field]=st.number_input(title,0.0,100.0,80.0,step=0.1,key='individual_'+field)
        submitted=st.form_submit_button('Run Risk Assessment',type='primary',use_container_width=True)
    if submitted:
        errors=validate_inputs(answers)
        if errors: st.error(' '.join(errors)); return
        render_result(predict_student(artifact,answers),student_id.strip() or 'Individual Assessment')

def dark_bar(data:pd.DataFrame,color:str='#35d6ff')->None:
    chart_data=data.reset_index(); category,value=chart_data.columns[:2]
    spec={'mark':{'type':'bar','cornerRadiusEnd':8,'color':color},'encoding':{'x':{'field':category,'type':'nominal','sort':'-y','axis':{'labelAngle':0,'title':None}},'y':{'field':value,'type':'quantitative','axis':{'title':None}},'tooltip':[{'field':category},{'field':value}]},'height':280,'config':{'background':'transparent','view':{'stroke':None},'axis':{'labelColor':'#a9bbcc','gridColor':'#26384e','domainColor':'#40556c','tickColor':'#40556c'}}}
    st.vega_lite_chart(chart_data,spec,use_container_width=True)

def batch_analysis(artifact:dict[str,Any])->None:
    st.markdown('<div class="page-header"><span>COHORT ANALYSIS</span><h1>Batch Analysis</h1><p>Analyze a CSV or Excel file of students using the same risk model and support rules.</p></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2,gap='medium'); c1.download_button('Download CSV Template',template_csv(),'student_batch_template.csv','text/csv',use_container_width=True,type='primary')
    with c2:
        c2.download_button('Download Excel Template',template_excel(),'student_batch_template.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True,type='primary')
    with st.expander('Using Google Forms'):
        st.markdown('Create the same questions → store responses in Google Sheets → export CSV or Excel (.xlsx) → upload here. Direct Google Sheets synchronization is future work.')
    st.markdown('<div class="upload-heading"><span>UPLOAD COHORT FILE</span><b>CSV or Excel · validated before analysis</b></div>',unsafe_allow_html=True)
    uploaded=st.file_uploader('Upload student data',type=['csv','xlsx'],label_visibility='collapsed')
    if uploaded is None: return
    try: frame=read_uploaded_table(uploaded)
    except Exception as error: st.error(f'Could not read uploaded file: {error}'); return
    errors=validate_batch_dataframe(frame)
    if not errors.empty:
        st.error('The file contains validation problems. No rows were analyzed.'); st.dataframe(errors,use_container_width=True,hide_index=True); return
    results=analyze_batch(artifact,frame); elevated=results['Risk Level'].str.startswith('Elevated')
    probability_column='Estimated Current Dropout-Status Probability'
    metric_cards([(str(len(results)),'Total Students','◎'),(str(int(elevated.sum())),'Elevated Estimate','▲'),(str(int((~elevated).sum())),'Lower Estimate','✓'),(f'{results[probability_column].mean():.1%}','Average Status Estimate','◉')])
    c1,c2=st.columns(2,gap='large')
    with c1:
        section_heading('Risk Distribution','COHORT OVERVIEW'); distribution=results['Risk Level'].value_counts().rename_axis('Risk Level').to_frame('Students'); dark_bar(distribution,'#7c6cff')
    with c2:
        section_heading('Most Common Warning Indicators','SUPPORT PRIORITIES'); indicators=results.loc[elevated,'Key Risk Indicators'].str.split(' | ',regex=False).explode(); indicators=indicators[indicators!='-']; dark_bar(indicators.value_counts().head(8).rename_axis('Indicator').to_frame('Students'),'#35d6ff')
    section_heading('Student Results','FILTER & REVIEW'); c1,c2,c3=st.columns(3); risk_filter=c1.selectbox('Risk Level',['All','Elevated','Lower']); semesters=['All']+sorted(x for x in results[SEMESTER_COLUMN].dropna().astype(str).unique() if x); semester_filter=c2.selectbox('Semester',semesters); search=c3.text_input('Search Anonymous Follow-Up ID',placeholder='Search ID')
    filtered=results
    if risk_filter!='All': filtered=filtered[filtered['Risk Level'].str.startswith(risk_filter)]
    if semester_filter!='All': filtered=filtered[filtered[SEMESTER_COLUMN].astype(str)==semester_filter]
    if search: filtered=filtered[filtered[ID_COLUMN].astype(str).str.contains(search,case=False,na=False)]
    display=filtered[[ID_COLUMN,SEMESTER_COLUMN,probability_column,'Risk Level','Main Indicator']].copy(); display[probability_column]=display[probability_column].map(lambda x:f'{x:.1%}')
    styled=display.style.apply(lambda row:['background-color: rgba(255,110,92,.10)' if str(row['Risk Level']).startswith('Elevated') else '' for _ in row],axis=1)
    st.dataframe(styled,use_container_width=True,hide_index=True)
    d1,d2=st.columns(2,gap='large')
    d1.download_button('↓ Download Analysis Results CSV',export_results(results),'student_batch_analysis.csv','text/csv',use_container_width=True)
    d2.download_button('↓ Download Analysis Results Excel',export_results_excel(results),'student_batch_analysis.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)

def about(artifact:dict[str,Any])->None:
    st.markdown('<div class="page-header"><span>PROJECT SUMMARY</span><h1>About Project</h1><p>A practical view of the system, model, and evaluation results.</p></div>',unsafe_allow_html=True)
    metrics=artifact.get('active_model_metrics',{})
    info_card('Project Purpose','This system helps identify students at higher dropout risk so institutions can take early support actions.')
    c1,c2=st.columns(2,gap='medium')
    with c1:
        info_card('Datasets','UCI Dataset: 3,630 resolved student records<br>Bangladesh Dataset: 251 university student responses')
    with c2:
        info_card('Model','Calibrated Logistic Regression Ensemble<br>Two Logistic Regression classifiers with a calibrated weighted combination.')
    section_heading('Model Architecture')
    render_model_architecture()
    section_heading('Performance','FIVE-FOLD CROSS-VALIDATION')
    render_performance_chart(metrics)
    section_heading('Classification Results','FIVE-FOLD CROSS-VALIDATION')
    render_confusion_matrix(metrics['confusion_matrix'])
    section_heading('Key Features')
    st.markdown('<div class="feature-list"><span>18 student input factors</span><span>Individual student assessment</span><span>Batch analysis</span><span>Risk indicators</span><span>Support recommendations</span></div>',unsafe_allow_html=True)

def bangladesh_data_collection()->None:
    st.markdown('<div class="page-header"><span>LOCAL RESEARCH DATA</span><h1>Bangladesh Data Collection</h1><p>Review and validate a Bangladesh-specific survey export with 18 predictors and one separate status label. This page does not calculate model risk scores.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="privacy-note"><b>Separate response-import workflow.</b> Imported survey files are validated and exported locally; they are not silently appended to training or sent into a prediction. For the trained Bangladesh classifier, Current Academic Status is used as the target only and never as an input. The questionnaire contains no identity, consent, university, date, or other metadata questions. Obtain and document any required consent separately before collecting or using responses.</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2,gap='large')
    with c1:
        info_card('Questionnaire schema','Version 3 contains exactly 18 predictor questions followed by one required Current Academic Status dropdown. It collects no consent, student ID, university name, dates, or other research metadata. The status field is excluded from every input feature set.')
        info_card('Parent occupations','The assessment form uses the original Bangladesh questionnaire categories, including Not Alive and Housewife. The UCI component uses ten harmonized features and does not learn parental occupation categories; no Portuguese-to-Bangladesh occupation crosswalk is applied. Housewife is represented in the mother encoder but not the father encoder, where it is handled as an unseen category. The separate Bangladesh research questionnaire retains its original categories unchanged.')
    with c2:
        info_card('Local fields','Attendance is 0–100% or blank. Household-income brackets are provisional survey categories, not official poverty definitions. Employment options describe student work. GPA fields are recorded without cross-country conversion.')
        info_card('Status label','Dropout (Discontinued Studies) maps to 1. Passed (Graduated) and Running (Currently Studying) map to 0, as specified. Running is only the status reported at submission and is not evidence of future completion or non-dropout.')

    with st.expander('Review the 18 predictor fields and response categories',expanded=False):
        rows=[]
        for field in BD_PREDICTOR_FIELDS:
            options=BD_CATEGORICAL_CHOICES.get(field,())
            rows.append({'Predictor':BD_FIELD_TITLES[field],'Type':'Categorical' if options else 'Numeric','Allowed values / notes':', '.join(options) if options else 'See versioned data dictionary; blank means unknown/not reported.'})
        rows.append({'Predictor':BD_STATUS_TITLE,'Type':'Target (not a feature)','Allowed values / notes':', '.join(BD_ACADEMIC_STATUS_OPTIONS)})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    template=questionnaire_template_frame()
    st.download_button('↓ Download Bangladesh survey CSV template',template.to_csv(index=False).encode('utf-8'),'bangladesh_university_survey_v3.csv','text/csv',use_container_width=True)
    st.caption('Run updateExistingBangladeshResearchForm() in your Google Apps Script project to update the existing form. It stops without changing anything if any responses already exist. It does not create, publish, or distribute a form.')

    section_heading('Import Google Forms responses','LOCAL VALIDATION ONLY')
    uploaded=st.file_uploader('Upload a Google Forms CSV or Excel (.xlsx) export',type=['csv','xlsx'],key='bangladesh_forms_upload')
    if uploaded is None:
        st.caption('CSV and Excel files are processed in memory. The clean export contains 18 predictor columns and the separate required academic-status label. Missing optional predictors remain blank; no imputation or model inference is performed.')
        return
    try:
        source=read_uploaded_table(uploaded)
        cleaned=clean_google_forms_responses(source)
        issues=validate_bangladesh_responses(source)
    except Exception as error:
        st.error(f'Could not validate the research export: {error}')
        return
    errors=issues.loc[issues['Severity'].eq('error')] if not issues.empty else issues
    warnings=issues.loc[issues['Severity'].eq('warning')] if not issues.empty else issues
    if not issues.empty:
        st.dataframe(issues,use_container_width=True,hide_index=True)
    if not errors.empty:
        st.error('Resolve the value errors before exporting this response file.')
        return
    c1,c2,c3=st.columns(3)
    c1.metric('Rows received',len(cleaned)); c2.metric('Predictor fields',len(BD_PREDICTOR_FIELDS)); c3.metric('Target labels',len(cleaned[BD_STATUS_FIELD]))
    if not warnings.empty:
        st.info('Warnings identify question columns absent from the uploaded export. Those fields are preserved as blank columns.')
    st.dataframe(cleaned,use_container_width=True,hide_index=True)
    csv_bytes=cleaned.to_csv(index=False).encode('utf-8')
    excel_bytes=bangladesh_research_excel(cleaned)
    d1,d2=st.columns(2)
    labels=academic_status_target(cleaned)
    st.caption('Target coding preview: Dropout = 1; Passed and Running = 0. Running is an observed current status, not confirmation of a future outcome.')
    st.dataframe(pd.DataFrame({'Current Academic Status':cleaned[BD_STATUS_FIELD],'dropout_target':labels}),use_container_width=True,hide_index=True)
    d1.download_button('↓ Download clean research CSV',csv_bytes,'bangladesh_university_responses_v3.csv','text/csv',use_container_width=True)
    d2.download_button('↓ Download clean research Excel',excel_bytes,'bangladesh_university_responses_v3.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)

def style()->None:
    st.markdown('''<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root{--bg:#07101e;--panel:#101d2e;--panel2:#0c1727;--line:rgba(133,174,214,.18);--cyan:#35d6ff;--violet:#8b7cff;--text:#f4f8fc;--muted:#91a5b9;--green:#38d996;--orange:#ff8a5b}
    html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif}.stApp{background:radial-gradient(circle at 80% -10%,rgba(52,87,170,.22),transparent 34%),radial-gradient(circle at 10% 20%,rgba(26,185,210,.08),transparent 28%),var(--bg);color:var(--text)}
    .block-container{max-width:1240px;padding:2.2rem 2.2rem 4rem}h1,h2,h3,h4,p,label{color:var(--text)}p,.stCaption{color:var(--muted)!important}.hero,.page-header{padding:2.2rem;border:1px solid var(--line);border-radius:24px;background:linear-gradient(135deg,rgba(18,38,61,.92),rgba(11,24,42,.78));box-shadow:0 24px 70px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.04);margin-bottom:1.4rem}.hero h1,.page-header h1{font-size:clamp(2.1rem,4.2vw,3.7rem);line-height:1.05;letter-spacing:-.05em;margin:.5rem 0;color:white}.hero h2{font-size:clamp(1.05rem,2vw,1.45rem);font-weight:500;color:#b9d7e8}.hero p,.page-header p{max-width:820px;font-size:1.02rem;margin-bottom:0}.hero-kicker,.page-header>span,.section-heading>span,.upload-heading span{color:var(--cyan);font-size:.72rem;font-weight:800;letter-spacing:.16em}.metric-card{position:relative;overflow:hidden;background:linear-gradient(145deg,rgba(19,35,55,.92),rgba(12,25,43,.82));border:1px solid var(--line);border-radius:18px;padding:1.1rem;box-shadow:0 15px 40px rgba(0,0,0,.22);height:140px;box-sizing:border-box;display:flex;flex-direction:column;justify-content:center;transition:.2s}.metric-card:hover{transform:translateY(-3px);border-color:rgba(53,214,255,.38)}.metric-card:after{content:'';position:absolute;width:90px;height:90px;border-radius:50%;background:var(--cyan);filter:blur(60px);opacity:.09;right:-20px;top:-20px}.metric-card strong{display:block;color:white;font-size:1.5rem;margin:.45rem 0 .2rem;line-height:1.15}.metric-card span{color:var(--muted);font-size:.8rem}.metric-icon{color:var(--cyan);font-size:1.05rem}.process-strip,.architecture{display:flex;align-items:center;justify-content:center;gap:.8rem;flex-wrap:wrap;margin:1.8rem 0;padding:1rem;border:1px solid var(--line);border-radius:16px;background:rgba(10,23,39,.72)}.process-strip span,.architecture span{color:#dcecf5;font-weight:650}.process-strip b,.architecture b{color:var(--cyan)}.action-copy{background:rgba(15,30,49,.75);border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.2rem;margin-bottom:.7rem}.action-copy b,.action-copy span{display:block}.action-copy span{color:var(--muted);font-size:.85rem;margin-top:.25rem}
    section[data-testid="stSidebar"]{background:linear-gradient(180deg,#091421,#07101c);border-right:1px solid var(--line)}section[data-testid="stSidebar"] h1{color:white;font-size:1.25rem}section[data-testid="stSidebar"] [role="radiogroup"] label{padding:.7rem .8rem;border-radius:10px;margin:.18rem 0;transition:.2s}section[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(53,214,255,.08)}
    div[data-testid="stForm"],div[data-testid="stExpander"],div[data-testid="stFileUploader"]{background:rgba(12,25,43,.78);border:1px solid var(--line)!important;border-radius:20px;padding:1rem;box-shadow:0 16px 45px rgba(0,0,0,.18)}.section-heading{margin:1.3rem 0 .7rem}.section-heading h3{margin:.15rem 0;color:white;font-size:1.22rem}div[data-baseweb="input"]>div,div[data-baseweb="select"]>div{background:#0b1727!important;border-color:#30445b!important;color:white!important;border-radius:10px!important}input{color:white!important}.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{border-radius:11px!important;border:1px solid rgba(53,214,255,.3)!important;background:linear-gradient(135deg,#167da3,#635ad9)!important;color:white!important;font-weight:700!important;min-height:2.9rem;box-shadow:0 8px 24px rgba(31,131,175,.18);transition:.2s!important}.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:linear-gradient(120deg,#08a8cf,#6257e8)!important;border-color:rgba(137,237,255,.72)!important;box-shadow:0 10px 30px rgba(42,190,232,.32),inset 0 1px 0 rgba(255,255,255,.28)!important}.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{transform:translateY(-2px);box-shadow:0 12px 30px rgba(53,214,255,.22)}
    .result-panel{border-radius:22px;padding:1.5rem;margin:1.5rem 0;border:1px solid var(--line);background:rgba(12,25,43,.86);box-shadow:0 18px 50px rgba(0,0,0,.25)}.result-top{display:flex;align-items:center;justify-content:space-between;gap:1rem}.result-label{display:block;color:var(--muted);font-size:.72rem;letter-spacing:.14em;font-weight:800}.result-panel strong{display:block;font-size:3.2rem;color:white}.status-badge{padding:.55rem .85rem;border-radius:999px;font-size:.78rem;font-weight:800;text-transform:uppercase}.elevated .status-badge{color:#ffd3c2;background:rgba(255,104,76,.16);border:1px solid rgba(255,104,76,.38)}.lower .status-badge{color:#adf4d1;background:rgba(56,217,150,.13);border:1px solid rgba(56,217,150,.34)}.risk-track{height:10px;background:#26364a;border-radius:99px;overflow:hidden;margin-top:1rem}.risk-track div{height:100%;background:linear-gradient(90deg,var(--cyan),var(--violet),var(--orange));border-radius:99px}.scale{display:flex;justify-content:space-between;color:#72879b;font-size:.7rem;margin-top:.4rem}.chip-grid,.support-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:.7rem;margin:.7rem 0 1.4rem}.indicator-chip,.support-card{display:flex;gap:.7rem;align-items:flex-start;border:1px solid var(--line);background:rgba(14,29,48,.8);border-radius:13px;padding:.9rem;color:#dce8f1}.indicator-chip i{color:var(--orange);font-style:normal;font-weight:800}.support-card i{color:var(--cyan);font-style:normal}.lower-note,.privacy-note{border:1px solid rgba(56,217,150,.28);background:rgba(56,217,150,.08);color:#c7f5df;padding:1rem;border-radius:13px;margin:1rem 0}.privacy-note{border-color:var(--line);background:rgba(53,214,255,.06);color:#bed9e8}.upload-heading{display:flex;justify-content:space-between;align-items:center;margin:1rem 0 .5rem}.upload-heading b{color:var(--muted);font-size:.8rem}.info-card{background:linear-gradient(145deg,rgba(17,33,53,.88),rgba(10,23,39,.82));border:1px solid var(--line);border-radius:16px;padding:1.2rem;margin-bottom:1rem;box-shadow:0 14px 38px rgba(0,0,0,.17)}.info-card h3{margin:0 0 .55rem;color:white}.info-card div{color:var(--muted);line-height:1.65}.feature-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:.7rem}.feature-list span{padding:.9rem 1rem;border-radius:12px;border:1px solid var(--line);background:rgba(14,29,48,.8);color:#dce8f1}[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden}[data-testid="stFileUploaderDropzone"]{background:#0b1727;border:1px dashed #36536d;border-radius:14px}[data-testid="stFileUploaderDropzone"] small{color:var(--muted)!important}[data-baseweb="popover"],[data-baseweb="menu"]{background:#0b1727!important;color:var(--text)!important}hr{border-color:var(--line)}
    @media(max-width:760px){.block-container{padding:1.2rem}.hero,.page-header{padding:1.35rem}.result-top{align-items:flex-start;flex-direction:column}.process-strip b,.architecture b{display:none}.metric-card{min-height:105px}.upload-heading{align-items:flex-start;flex-direction:column;gap:.25rem}}
    </style>''',unsafe_allow_html=True)
    st.markdown('''<style>
    .performance-chart{padding:1rem 1.15rem;border:1px solid var(--line);border-radius:17px;background:linear-gradient(145deg,rgba(17,33,53,.86),rgba(10,23,39,.78))}
    .performance-row{display:grid;grid-template-columns:100px minmax(80px,1fr) 58px;align-items:center;gap:1rem;padding:.52rem 0;color:#dce8f1;font-size:.88rem}
    .performance-row b{text-align:right;color:white;font-variant-numeric:tabular-nums}.performance-track{height:12px;border-radius:99px;background:#23364b;overflow:hidden}.performance-track i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#20c7e8,#8274ff);box-shadow:0 0 14px rgba(53,214,255,.24)}
    .support-metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem;margin:.75rem 0 1rem}.support-metric{display:flex;justify-content:space-between;align-items:center;padding:.85rem 1rem;border:1px solid var(--line);border-radius:14px;background:rgba(12,25,43,.82);color:var(--muted)}.support-metric strong{color:#cbdcff;font-size:1.08rem}
    .matrix-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:17px;background:rgba(12,25,43,.82)}.confusion-matrix{width:100%;border-collapse:separate;border-spacing:6px;padding:.55rem;min-width:540px}.confusion-matrix th{color:#b9c9d9;font-size:.78rem;font-weight:650;text-align:center;padding:.6rem}.confusion-matrix tbody th{text-align:left;white-space:nowrap}.matrix-cell{height:78px;text-align:center;border:1px solid;border-radius:12px;min-width:150px}.matrix-cell b{display:block;font-size:1.65rem;line-height:1.2;color:white;font-variant-numeric:tabular-nums}.matrix-cell small{font-size:.7rem;color:#d6e3ed}.matrix-cell.correct{background:linear-gradient(145deg,rgba(24,143,126,.4),rgba(25,111,121,.22));border-color:rgba(61,221,184,.42)}.matrix-cell.mistake{background:linear-gradient(145deg,rgba(179,83,64,.38),rgba(136,59,74,.22));border-color:rgba(255,137,114,.42)}
    .matrix-summary{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem;margin:.8rem 0}.matrix-summary>div{display:flex;align-items:baseline;justify-content:space-between;gap:.6rem;padding:.8rem 1rem;border:1px solid var(--line);border-radius:14px;background:rgba(12,25,43,.82)}.matrix-summary strong{color:white;font-size:1.3rem}.matrix-summary strong small{font-size:.85rem;color:var(--muted)}.matrix-summary span{color:var(--muted);font-size:.82rem}.matrix-legend{font-size:.75rem!important;line-height:1.55;color:#9bb0c4!important;margin:.4rem 0 1rem}
    .model-architecture{padding:1rem;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,rgba(17,33,53,.86),rgba(10,23,39,.78))}.model-branches{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem}.model-branch{display:flex;flex-direction:column;gap:.45rem;min-height:112px;padding:1rem;border:1px solid rgba(53,214,255,.22);border-radius:14px;background:rgba(10,24,42,.78)}.model-branch span,.model-branch small{color:var(--muted);font-size:.78rem}.model-branch b{color:#eaf7ff;font-size:.94rem}.architecture-merge{text-align:center;color:var(--cyan);font-size:.82rem;font-weight:700;padding:.85rem .3rem .35rem}.architecture-steps{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:.65rem}.architecture-steps span,.architecture-steps strong{padding:.65rem .85rem;border:1px solid var(--line);border-radius:11px;background:rgba(11,25,43,.92);color:#dce8f1;font-size:.8rem}.architecture-steps strong{color:#a9f1d3;border-color:rgba(56,217,150,.34)}.architecture-steps b{color:var(--cyan)}
    @media(max-width:640px){.performance-row{grid-template-columns:78px minmax(50px,1fr) 48px;gap:.6rem;font-size:.78rem}.performance-track{height:10px}.matrix-summary{grid-template-columns:1fr}.model-branches{grid-template-columns:1fr}.architecture-steps{flex-direction:column}.architecture-steps b{transform:rotate(90deg)}}
    </style>''',unsafe_allow_html=True)

def main()->None:
    st.set_page_config(page_title='Student Dropout Early Warning System',page_icon='🎓',layout='wide',initial_sidebar_state='expanded'); style(); artifact=load_model_artifact()
    st.sidebar.markdown('### ◈ Early Warning'); st.sidebar.caption('STUDENT SUCCESS INTELLIGENCE'); page=st.sidebar.radio('Navigate',APP_PAGES,key='page',label_visibility='collapsed'); st.sidebar.markdown('---'); st.sidebar.caption('Bangladesh research prototype · 18 predictors')
    if page=='Dashboard': dashboard(artifact)
    elif page=='Individual Assessment': individual_assessment(artifact)
    elif page=='Batch Analysis': batch_analysis(artifact)
    elif page=='About Project': about(artifact)
if __name__=='__main__': main()
