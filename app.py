"""Local Student Dropout Early Warning System."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import io
import joblib
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/'models'/'uci_sem1'/'dropout_early_warning_model.joblib'
METADATA_PATH=ROOT/'models'/'uci_sem1'/'model_metadata.json'
THRESHOLD=None
SOURCE_SEMESTER_GRADE_MAX=18.875
FORM_FIELDS=['previous_academic_gpa_normalized','Age at enrollment',"Mother's occupation","Father's occupation","Mother's qualification","Father's qualification",'Gender','Marital status','Scholarship holder','Debtor','Tuition fees up to date','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)','Curricular units 1st sem (grade)','Curricular units 1st sem (approved)']
USER_FIELDS=['HSC / Equivalent GPA','Age at Enrollment',"Mother's Occupation","Father's Occupation","Mother's Education","Father's Education",'Gender','Marital Status','Scholarship Holder','Debtor','Tuition Fees Up to Date','Courses Enrolled','Evaluations Completed','Courses Passed','Semester GPA']
ID_COLUMN='Anonymous Follow-Up ID'
SEMESTER_COLUMN='Current Semester / Academic Year'
BATCH_COLUMNS=[ID_COLUMN,SEMESTER_COLUMN,*USER_FIELDS]
RESULT_COLUMNS=[ID_COLUMN,SEMESTER_COLUMN,'Estimated Dropout Risk','Risk Level','Main Indicator','Key Risk Indicators','Suggested Institute Support']
OCCUPATIONS={0:'Student',1:'Legislative/executive representatives, directors and managers',2:'Intellectual and scientific specialists',3:'Intermediate-level technicians and professions',4:'Administrative staff',5:'Personal services, security and sales workers',6:'Skilled agriculture, fisheries and forestry workers',7:'Skilled industry, construction and craft workers',8:'Machine operators and assembly workers',9:'Unskilled workers',10:'Armed Forces',90:'Other situation',99:'Unspecified',101:'Armed Forces officers',102:'Armed Forces sergeants',103:'Other Armed Forces personnel',112:'Administrative and commercial services directors',114:'Hotel, catering, trade and other service directors',121:'Physical sciences, mathematics and engineering specialists',122:'Health professionals',123:'Teachers',124:'Finance, administration and commercial specialists',125:'Information and communication technology specialists',131:'Intermediate science and engineering technicians',132:'Intermediate health technicians',134:'Intermediate legal, social, cultural and sports services',135:'Information and communication technology technicians',141:'Office, secretarial and data-processing workers',143:'Data, accounting, statistics and financial operators',144:'Other administrative support staff',151:'Personal service workers',152:'Sellers',153:'Personal care workers',154:'Protection and security personnel',161:'Market-oriented farmers and skilled agricultural workers',163:'Subsistence farmers, fishers, hunters and gatherers',171:'Skilled construction workers (except electricians)',172:'Skilled metallurgy and metalworking workers',173:'Printing, precision, jewellery and craft workers',174:'Skilled electrical and electronics workers',175:'Food, wood, clothing and other craft workers',181:'Fixed-plant and machine operators',182:'Assembly workers',183:'Vehicle drivers and mobile-equipment operators',191:'Cleaning workers',192:'Unskilled agriculture, fisheries and forestry workers',193:'Unskilled extractive, construction, manufacturing and transport workers',194:'Meal preparation assistants',195:'Street vendors and service providers'}
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
    artifact=joblib.load(MODEL_PATH)
    if artifact.get('raw_input_features')!=FORM_FIELDS or len(artifact.get('features',[]))!=15:
        raise RuntimeError('The active model does not match the required 15-input schema. Run: python -m src.train_uci_sem1_15')
    return artifact

def semester_gpa_to_source_grade(value:float)->float: return float(value)/4.0*SOURCE_SEMESTER_GRADE_MAX

def make_input_frame(values:dict[str,Any])->pd.DataFrame:
    if set(values)!=set(FORM_FIELDS): raise ValueError('Answers do not match the required 15-field form.')
    return pd.DataFrame([{field:values[field] for field in FORM_FIELDS}])

def validate_inputs(values:dict[str,Any])->list[str]:
    return ['Courses Passed cannot be greater than Courses Enrolled.'] if values['Curricular units 1st sem (approved)']>values['Curricular units 1st sem (enrolled)'] else []

def dropout_probability(artifact:dict[str,Any],values:dict[str,Any])->float:
    pipeline=artifact['pipeline']; index=list(pipeline.classes_).index(1)
    if artifact.get('raw_input_features')!=FORM_FIELDS:
        raise ValueError('The saved model does not support this 15-field assessment. Retrain it first.')
    return float(pipeline.predict_proba(make_input_frame(values))[0,index])

def risk_label(probability:float,threshold:float|None=None)->str:
    if threshold is None: threshold=load_model_artifact()['threshold']
    return 'Elevated Estimated Risk' if probability>=threshold else 'Lower Estimated Risk'

def support_guidance(*,semester_gpa:float,enrolled:int,evaluations:int,passed:int,debtor:bool,tuition_up_to_date:bool,scholarship:bool,previous_gpa:float)->tuple[list[str],list[str]]:
    pass_rate=passed/enrolled if enrolled else 0.0; evaluation_rate=evaluations/enrolled if enrolled else 0.0; candidates=[]
    if semester_gpa<2.0: candidates.append((10,f'Low recent academic performance — Semester GPA: {semester_gpa:.2f} / 4.00',['Academic advising','Tutoring / study support']))
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
    if value in OCCUPATION_BY_LABEL: return OCCUPATION_BY_LABEL[value]
    try: code=int(float(value))
    except (TypeError,ValueError): raise ValueError('is not a supported occupation label')
    if code not in OCCUPATIONS: raise ValueError('is not a supported occupation label')
    return code

def prepare_student(record:dict[str,Any])->dict[str,Any]:
    return {'previous_academic_gpa_normalized':float(record['HSC / Equivalent GPA']),'Age at enrollment':int(record['Age at Enrollment']),"Mother's occupation":occupation_code(record["Mother's Occupation"]),"Father's occupation":occupation_code(record["Father's Occupation"]),"Mother's qualification":education_code(record["Mother's Education"]),"Father's qualification":education_code(record["Father's Education"]),'Gender':gender_code(record['Gender']),'Marital status':marital_code(record['Marital Status']),'Scholarship holder':int(parse_boolean(record['Scholarship Holder'])),'Debtor':int(parse_boolean(record['Debtor'])),'Tuition fees up to date':int(parse_boolean(record['Tuition Fees Up to Date'])),'Curricular units 1st sem (enrolled)':int(record['Courses Enrolled']),'Curricular units 1st sem (evaluations)':int(record['Evaluations Completed']),'Curricular units 1st sem (approved)':int(record['Courses Passed']),'Curricular units 1st sem (grade)':semester_gpa_to_source_grade(float(record['Semester GPA']))}

def predict_student(artifact:dict[str,Any],record:dict[str,Any])->dict[str,Any]:
    values=prepare_student(record); probability=dropout_probability(artifact,values); label=risk_label(probability,artifact['threshold'])
    indicators,supports=support_guidance(semester_gpa=float(record['Semester GPA']),enrolled=int(record['Courses Enrolled']),evaluations=int(record['Evaluations Completed']),passed=int(record['Courses Passed']),debtor=bool(values['Debtor']),tuition_up_to_date=bool(values['Tuition fees up to date']),scholarship=bool(values['Scholarship holder']),previous_gpa=float(record['HSC / Equivalent GPA']))
    return {'probability':probability,'risk_level':label,'threshold':artifact['threshold'],'indicators':indicators,'supports':supports}

def validate_batch_dataframe(frame:pd.DataFrame)->pd.DataFrame:
    missing=[column for column in USER_FIELDS if column not in frame.columns]
    if missing: return pd.DataFrame([{'Row':'File','Issue':'Missing required columns: '+', '.join(missing)}])
    issues=[]
    for position,(_,row) in enumerate(frame.iterrows(),start=2):
        def issue(message:str): issues.append({'Row':position,'Issue':message})
        try:
            hsc=float(row['HSC / Equivalent GPA'])
            if not 2.5<=hsc<=5.0: issue('HSC / Equivalent GPA must be between 2.50 and 5.00.')
        except (TypeError,ValueError): issue('HSC / Equivalent GPA must be numeric.')
        try:
            gpa=float(row['Semester GPA'])
            if not 0<=gpa<=4: issue('Semester GPA must be between 0.00 and 4.00.')
        except (TypeError,ValueError): issue('Semester GPA must be numeric.')
        for column,minimum,maximum in [('Age at Enrollment',17,70),('Courses Enrolled',0,26),('Evaluations Completed',0,None),('Courses Passed',0,26)]:
            try:
                value=float(row[column])
                if value<minimum or (maximum is not None and value>maximum) or not value.is_integer(): issue(f'{column} must be a whole number in the supported range.')
            except (TypeError,ValueError): issue(f'{column} must be numeric.')
        try:
            if float(row['Courses Passed'])>float(row['Courses Enrolled']): issue('Courses Passed cannot be greater than Courses Enrolled.')
        except (TypeError,ValueError): pass
        for column in ['Scholarship Holder','Debtor','Tuition Fees Up to Date']:
            try: parse_boolean(row[column])
            except ValueError as error: issue(f'{column} {error}.')
        for column in ["Mother's Occupation","Father's Occupation"]:
            try: occupation_code(row[column])
            except ValueError as error: issue(f'{column} {error}.')
    for position,(_,row) in enumerate(frame.iterrows(),start=2):
        for column in ["Mother's Education","Father's Education"]:
            try: education_code(row[column])
            except ValueError as error: issues.append({'Row':position,'Issue':f'{column} {error}.'})
        try: marital_code(row['Marital Status'])
        except ValueError as error: issues.append({'Row':position,'Issue':f'Marital Status {error}.'})
        try: gender_code(row['Gender'])
        except ValueError as error: issues.append({'Row':position,'Issue':f'Gender {error}.'})
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
        rows.append({**record,ID_COLUMN:metadata_text(record.get(ID_COLUMN),f'Student {position}'),SEMESTER_COLUMN:metadata_text(record.get(SEMESTER_COLUMN)), 'Estimated Dropout Risk':result['probability'],'Risk Level':result['risk_level'],'Main Indicator':indicators[0] if indicators else '-','Key Risk Indicators':' | '.join(indicators) if indicators else '-','Suggested Institute Support':' | '.join(result['supports']) if indicators else '-','Warning Type':indicator_group(indicators[0]) if indicators else '-'})
    return pd.DataFrame(rows)

def export_results(frame:pd.DataFrame)->bytes:
    columns=[*BATCH_COLUMNS,*RESULT_COLUMNS[2:]]
    return frame[[column for column in columns if column in frame.columns]].to_csv(index=False).encode('utf-8')

def template_frame()->pd.DataFrame:
    example={ID_COLUMN:'EXAMPLE-001',SEMESTER_COLUMN:'3rd Semester','HSC / Equivalent GPA':4.20,'Age at Enrollment':19,"Mother's Occupation":'Administrative staff',"Father's Occupation":'Unskilled workers','Scholarship Holder':'No','Debtor':'No','Tuition Fees Up to Date':'Yes','Courses Enrolled':6,'Evaluations Completed':6,'Courses Passed':5,'Semester GPA':3.25,"Mother's Education":'Secondary education (12th year)',"Father's Education":'Secondary education (12th year)','Gender':'Female','Marital Status':'Single'}
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

def dataframe_to_excel(frame:pd.DataFrame,sheet_name:str)->bytes:
    buffer=io.BytesIO()
    with pd.ExcelWriter(buffer,engine='openpyxl') as writer:
        frame.to_excel(writer,index=False,sheet_name=sheet_name)
    return buffer.getvalue()

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

def dashboard()->None:
    st.markdown('''<div class="hero"><div class="hero-kicker">ACADEMIC INTELLIGENCE · EARLY WARNING</div><h1>Student Dropout Early Warning System</h1><h2>AI-powered student support dashboard for early academic intervention</h2><p>Identify potentially at-risk students and coordinate earlier support using academic, financial, and background information.</p></div>''',unsafe_allow_html=True)
    metric_cards([('Random Forest','ML Model','◈'),('15','Student Factors','◆'),('94.1%','Holdout ROC-AUC','◉'),('85.9%','Holdout Recall','↗')])
    st.markdown('<div class="process-strip"><span>Student Data</span><b>→</b><span>Risk Analysis</span><b>→</b><span>Warning Indicators</span><b>→</b><span>Support Action</span></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2,gap='large')
    c1.markdown('<div class="action-copy"><b>Individual screening</b><span>Review one student with an immediate support summary.</span></div>',unsafe_allow_html=True)
    c1.button('Assess One Student',type='primary',use_container_width=True,key='dashboard_assess_one',on_click=navigate,args=('Individual Assessment',))
    c2.markdown('<div class="action-copy"><b>Department overview</b><span>Upload a cohort file and identify support priorities.</span></div>',unsafe_allow_html=True)
    c2.button('Analyze a Student Batch',use_container_width=True,key='dashboard_analyze_batch',on_click=navigate,args=('Batch Analysis',))

def render_result(result:dict[str,Any],student_id:str='Individual Assessment')->None:
    elevated=result['risk_level'].startswith('Elevated'); status='elevated' if elevated else 'lower'; width=max(2,min(100,result['probability']*100))
    st.markdown(f'''<div class="result-panel {status}"><div class="result-top"><div><span class="result-label">ESTIMATED DROPOUT RISK</span><strong>{result["probability"]:.1%}</strong></div><div class="status-badge">{result["risk_level"]}</div></div><div class="risk-track"><div style="width:{width:.1f}%"></div></div><div class="scale"><span>Lower risk</span><span>Threshold {result["threshold"]:.0%}</span><span>Higher risk</span></div></div>''',unsafe_allow_html=True)
    if elevated:
        section_heading('Key Risk Indicators','INTERPRETATION')
        st.caption('These indicators are based on the entered information and are not confirmed causes of dropout.')
        indicators=result['indicators'] or ['No single actionable warning met the rule thresholds.']
        st.markdown('<div class="chip-grid">'+''.join(f'<div class="indicator-chip"><i>!</i><span>{item}</span></div>' for item in indicators)+'</div>',unsafe_allow_html=True)
        section_heading('Possible Institute Support','RECOMMENDATIONS')
        supports=result['supports'] or ['Continue an individual academic-support review']
        st.markdown('<div class="support-grid">'+''.join(f'<div class="support-card"><i>→</i><span>{item}</span></div>' for item in supports)+'</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="lower-note">✓ No major immediate warning indicators were identified. Continue normal academic monitoring.</div>',unsafe_allow_html=True)
    report='\n'.join([f'Student: {student_id}',f'Estimated Dropout Risk: {result["probability"]:.1%}',f'Risk Level: {result["risk_level"]}','Key Risk Indicators: '+('; '.join(result['indicators']) or 'None'),'Suggested Institute Support: '+('; '.join(result['supports']) or 'Normal academic monitoring')])
    st.download_button('↓ Download Assessment',report,file_name='student_assessment.txt',mime='text/plain')

def individual_assessment(artifact:dict[str,Any])->None:
    st.markdown('<div class="page-header"><span>INDIVIDUAL SCREENING</span><h1>Individual Assessment</h1><p>Use information from the student’s most recently completed semester.</p></div>',unsafe_allow_html=True)
    with st.form('assessment_form'):
        student_id=st.text_input('Anonymous Student ID (optional)',placeholder='e.g. STUDENT-001')
        section_heading('Student Background','01'); c1,c2=st.columns(2); previous=c1.number_input('HSC / Equivalent GPA',2.50,5.00,3.50,step=0.01); age=c2.number_input('Age at Enrollment',17,70,19)
        section_heading('Family Background','02'); c1,c2=st.columns(2); labels=list(OCCUPATION_BY_LABEL); mother=c1.selectbox("Mother’s Occupation",labels,index=labels.index('Unskilled workers')); father=c2.selectbox("Father’s Occupation",labels,index=labels.index('Unskilled workers'))
        c1,c2=st.columns(2); education_labels=list(EDUCATION_BY_LABEL); mother_education=c1.selectbox("Mother’s Education",education_labels); father_education=c2.selectbox("Father’s Education",education_labels)
        gender=st.selectbox('Gender',list(GENDERS)); marital=st.selectbox('Marital Status',list(MARITAL_STATUSES))
        section_heading('Financial Status','03'); c1,c2,c3=st.columns(3); scholarship=c1.selectbox('Scholarship Holder',['No','Yes']); debtor=c2.selectbox('Debtor',['No','Yes']); tuition=c3.selectbox('Tuition Fees Up to Date',['No','Yes'])
        section_heading('Recent Semester Performance','04'); c1,c2,c3,c4=st.columns(4); enrolled=c1.number_input('Courses Enrolled',0,26,6); evaluations=c2.number_input('Evaluations Completed',0,45,8); passed=c3.number_input('Courses Passed',0,26,5); semester_gpa=c4.number_input('Semester GPA',0.00,4.00,3.00,step=0.01)
        submitted=st.form_submit_button('Run Risk Assessment',type='primary',use_container_width=True)
    if submitted:
        record={'HSC / Equivalent GPA':previous,'Age at Enrollment':age,"Mother's Occupation":mother,"Father's Occupation":father,"Mother's Education":mother_education,"Father's Education":father_education,'Gender':gender,'Marital Status':marital,'Scholarship Holder':scholarship,'Debtor':debtor,'Tuition Fees Up to Date':tuition,'Courses Enrolled':enrolled,'Evaluations Completed':evaluations,'Courses Passed':passed,'Semester GPA':semester_gpa}
        if passed>enrolled: st.error('Courses Passed cannot be greater than Courses Enrolled.'); return
        render_result(predict_student(artifact,record),student_id.strip() or 'Individual Assessment')

def dark_bar(data:pd.DataFrame,color:str='#35d6ff')->None:
    chart_data=data.reset_index(); category,value=chart_data.columns[:2]
    spec={'mark':{'type':'bar','cornerRadiusEnd':8,'color':color},'encoding':{'x':{'field':category,'type':'nominal','sort':'-y','axis':{'labelAngle':0,'title':None}},'y':{'field':value,'type':'quantitative','axis':{'title':None}},'tooltip':[{'field':category},{'field':value}]},'height':280,'config':{'background':'transparent','view':{'stroke':None},'axis':{'labelColor':'#a9bbcc','gridColor':'#26384e','domainColor':'#40556c','tickColor':'#40556c'}}}
    st.vega_lite_chart(chart_data,spec,use_container_width=True)

def batch_analysis(artifact:dict[str,Any])->None:
    st.markdown('<div class="page-header"><span>COHORT INTELLIGENCE</span><h1>Batch Analysis</h1><p>Analyze a CSV or Excel cohort using the same validated model and support rules.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="privacy-note"><b>Privacy first.</b> Use anonymous student IDs. Do not upload names, phone numbers, email addresses, or unnecessary personal information. Files are processed in memory only.</div>',unsafe_allow_html=True)
    c1,c2=st.columns([1,2]); c1.download_button('↓ Download CSV Template',template_csv(),'student_batch_template.csv','text/csv',use_container_width=True)
    with c2:
        c2.download_button('↓ Download Excel Template',template_excel(),'student_batch_template.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)
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
    metric_cards([(str(len(results)),'Total Students','◎'),(str(int(elevated.sum())),'Elevated Risk','▲'),(str(int((~elevated).sum())),'Lower Risk','✓'),(f'{results["Estimated Dropout Risk"].mean():.1%}','Average Estimated Risk','◉')])
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
    display=filtered[[ID_COLUMN,SEMESTER_COLUMN,'Estimated Dropout Risk','Risk Level','Main Indicator']].copy(); display['Estimated Dropout Risk']=display['Estimated Dropout Risk'].map(lambda x:f'{x:.1%}')
    styled=display.style.apply(lambda row:['background-color: rgba(255,110,92,.10)' if str(row['Risk Level']).startswith('Elevated') else '' for _ in row],axis=1)
    st.dataframe(styled,use_container_width=True,hide_index=True)
    d1,d2=st.columns(2,gap='large')
    d1.download_button('↓ Download Analysis Results CSV',export_results(results),'student_batch_analysis.csv','text/csv',use_container_width=True)
    d2.download_button('↓ Download Analysis Results Excel',export_results_excel(results),'student_batch_analysis.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)

def about()->None:
    st.markdown('<div class="page-header"><span>PROJECT OVERVIEW</span><h1>About Project</h1><p>A transparent academic prototype connecting risk estimation with practical student support.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="architecture"><span>Student Inputs</span><b>→</b><span>Random Forest</span><b>→</b><span>Risk Estimate</span><b>→</b><span>Support Guidance</span></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2,gap='large')
    with c1:
        info_card('Purpose','Identify students who may benefit from timely academic or financial follow-up.')
        info_card('Dataset','UCI Predict Students’ Dropout and Academic Success · 4,424 original records · 3,630 resolved outcomes.')
        info_card('Model','Random Forest · 15 student inputs · end-of-Semester-1 prediction · threshold 0.52 selected from training-only OOF predictions.')
    with c2:
        info_card('Performance','Previously consulted holdout: accuracy 87.9%, precision 83.6%, recall 85.9%, F1 84.7%, ROC-AUC 94.1%, PR-AUC 93.1%. Confusion matrix: TN 394 · FP 48 · FN 40 · TP 244.')
        info_card('How Risk Indicators Work','A separate deterministic layer summarizes relevant academic and financial warnings. They are not causal explanations.')
        info_card('Local Data Collection','Anonymous campus records can support local pattern analysis, later validation, and future retraining once outcomes are linked.')
    info_card('Limitations','Training data originate from Portuguese higher education and are not externally validated in Bangladesh. Source academic variables describe first-semester performance. Local longitudinal outcomes are required before retraining. Predictions support—not replace—human judgement.')

def style()->None:
    st.markdown('''<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root{--bg:#07101e;--panel:#101d2e;--panel2:#0c1727;--line:rgba(133,174,214,.18);--cyan:#35d6ff;--violet:#8b7cff;--text:#f4f8fc;--muted:#91a5b9;--green:#38d996;--orange:#ff8a5b}
    html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif}.stApp{background:radial-gradient(circle at 80% -10%,rgba(52,87,170,.22),transparent 34%),radial-gradient(circle at 10% 20%,rgba(26,185,210,.08),transparent 28%),var(--bg);color:var(--text)}
    .block-container{max-width:1240px;padding:2.2rem 2.2rem 4rem}h1,h2,h3,h4,p,label{color:var(--text)}p,.stCaption{color:var(--muted)!important}.hero,.page-header{padding:2.2rem;border:1px solid var(--line);border-radius:24px;background:linear-gradient(135deg,rgba(18,38,61,.92),rgba(11,24,42,.78));box-shadow:0 24px 70px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.04);margin-bottom:1.4rem}.hero h1,.page-header h1{font-size:clamp(2.1rem,4.2vw,3.7rem);line-height:1.05;letter-spacing:-.05em;margin:.5rem 0;color:white}.hero h2{font-size:clamp(1.05rem,2vw,1.45rem);font-weight:500;color:#b9d7e8}.hero p,.page-header p{max-width:820px;font-size:1.02rem;margin-bottom:0}.hero-kicker,.page-header>span,.section-heading>span,.upload-heading span{color:var(--cyan);font-size:.72rem;font-weight:800;letter-spacing:.16em}.metric-card{position:relative;overflow:hidden;background:linear-gradient(145deg,rgba(19,35,55,.92),rgba(12,25,43,.82));border:1px solid var(--line);border-radius:18px;padding:1.2rem;box-shadow:0 15px 40px rgba(0,0,0,.22);min-height:130px;transition:.2s}.metric-card:hover{transform:translateY(-3px);border-color:rgba(53,214,255,.38)}.metric-card:after{content:'';position:absolute;width:90px;height:90px;border-radius:50%;background:var(--cyan);filter:blur(60px);opacity:.09;right:-20px;top:-20px}.metric-card strong{display:block;color:white;font-size:1.65rem;margin:.55rem 0 .25rem}.metric-card span{color:var(--muted);font-size:.85rem}.metric-icon{color:var(--cyan);font-size:1.15rem}.process-strip,.architecture{display:flex;align-items:center;justify-content:center;gap:.8rem;flex-wrap:wrap;margin:1.8rem 0;padding:1rem;border:1px solid var(--line);border-radius:16px;background:rgba(10,23,39,.72)}.process-strip span,.architecture span{color:#dcecf5;font-weight:650}.process-strip b,.architecture b{color:var(--cyan)}.action-copy{background:rgba(15,30,49,.75);border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.2rem;margin-bottom:.7rem}.action-copy b,.action-copy span{display:block}.action-copy span{color:var(--muted);font-size:.85rem;margin-top:.25rem}
    section[data-testid="stSidebar"]{background:linear-gradient(180deg,#091421,#07101c);border-right:1px solid var(--line)}section[data-testid="stSidebar"] h1{color:white;font-size:1.25rem}section[data-testid="stSidebar"] [role="radiogroup"] label{padding:.7rem .8rem;border-radius:10px;margin:.18rem 0;transition:.2s}section[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(53,214,255,.08)}
    div[data-testid="stForm"],div[data-testid="stExpander"],div[data-testid="stFileUploader"]{background:rgba(12,25,43,.78);border:1px solid var(--line)!important;border-radius:20px;padding:1rem;box-shadow:0 16px 45px rgba(0,0,0,.18)}.section-heading{margin:1.3rem 0 .7rem}.section-heading h3{margin:.15rem 0;color:white;font-size:1.22rem}div[data-baseweb="input"]>div,div[data-baseweb="select"]>div{background:#0b1727!important;border-color:#30445b!important;color:white!important;border-radius:10px!important}input{color:white!important}.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{border-radius:11px!important;border:1px solid rgba(53,214,255,.3)!important;background:linear-gradient(135deg,#167da3,#635ad9)!important;color:white!important;font-weight:700!important;min-height:2.9rem;box-shadow:0 8px 24px rgba(31,131,175,.18);transition:.2s!important}.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{transform:translateY(-2px);box-shadow:0 12px 30px rgba(53,214,255,.22)}
    .result-panel{border-radius:22px;padding:1.5rem;margin:1.5rem 0;border:1px solid var(--line);background:rgba(12,25,43,.86);box-shadow:0 18px 50px rgba(0,0,0,.25)}.result-top{display:flex;align-items:center;justify-content:space-between;gap:1rem}.result-label{display:block;color:var(--muted);font-size:.72rem;letter-spacing:.14em;font-weight:800}.result-panel strong{display:block;font-size:3.2rem;color:white}.status-badge{padding:.55rem .85rem;border-radius:999px;font-size:.78rem;font-weight:800;text-transform:uppercase}.elevated .status-badge{color:#ffd3c2;background:rgba(255,104,76,.16);border:1px solid rgba(255,104,76,.38)}.lower .status-badge{color:#adf4d1;background:rgba(56,217,150,.13);border:1px solid rgba(56,217,150,.34)}.risk-track{height:10px;background:#26364a;border-radius:99px;overflow:hidden;margin-top:1rem}.risk-track div{height:100%;background:linear-gradient(90deg,var(--cyan),var(--violet),var(--orange));border-radius:99px}.scale{display:flex;justify-content:space-between;color:#72879b;font-size:.7rem;margin-top:.4rem}.chip-grid,.support-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:.7rem;margin:.7rem 0 1.4rem}.indicator-chip,.support-card{display:flex;gap:.7rem;align-items:flex-start;border:1px solid var(--line);background:rgba(14,29,48,.8);border-radius:13px;padding:.9rem;color:#dce8f1}.indicator-chip i{color:var(--orange);font-style:normal;font-weight:800}.support-card i{color:var(--cyan);font-style:normal}.lower-note,.privacy-note{border:1px solid rgba(56,217,150,.28);background:rgba(56,217,150,.08);color:#c7f5df;padding:1rem;border-radius:13px;margin:1rem 0}.privacy-note{border-color:var(--line);background:rgba(53,214,255,.06);color:#bed9e8}.upload-heading{display:flex;justify-content:space-between;align-items:center;margin:1rem 0 .5rem}.upload-heading b{color:var(--muted);font-size:.8rem}.info-card{background:linear-gradient(145deg,rgba(17,33,53,.88),rgba(10,23,39,.82));border:1px solid var(--line);border-radius:16px;padding:1.2rem;margin-bottom:1rem;box-shadow:0 14px 38px rgba(0,0,0,.17)}.info-card h3{margin:0 0 .55rem;color:white}.info-card div{color:var(--muted);line-height:1.65}[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden}[data-testid="stFileUploaderDropzone"]{background:#0b1727;border:1px dashed #36536d;border-radius:14px}[data-testid="stFileUploaderDropzone"] small{color:var(--muted)!important}[data-baseweb="popover"],[data-baseweb="menu"]{background:#0b1727!important;color:var(--text)!important}hr{border-color:var(--line)}
    @media(max-width:760px){.block-container{padding:1.2rem}.hero,.page-header{padding:1.35rem}.result-top{align-items:flex-start;flex-direction:column}.process-strip b,.architecture b{display:none}.metric-card{min-height:105px}.upload-heading{align-items:flex-start;flex-direction:column;gap:.25rem}}
    </style>''',unsafe_allow_html=True)

def main()->None:
    st.set_page_config(page_title='Student Dropout Early Warning System',page_icon='🎓',layout='wide',initial_sidebar_state='expanded'); style(); artifact=load_model_artifact()
    pages=['Dashboard','Individual Assessment','Batch Analysis','About Project']; st.sidebar.markdown('### ◈ Early Warning'); st.sidebar.caption('STUDENT SUCCESS INTELLIGENCE'); page=st.sidebar.radio('Navigate',pages,key='page',label_visibility='collapsed'); st.sidebar.markdown('---'); st.sidebar.caption('Local academic prototype · 15-input Random Forest')
    if page=='Dashboard': dashboard()
    elif page=='Individual Assessment': individual_assessment(artifact)
    elif page=='Batch Analysis': batch_analysis(artifact)
    else: about()
if __name__=='__main__': main()
