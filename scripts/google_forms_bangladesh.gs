/**
 * Updates the existing Bangladesh questionnaire to the v3 form with 18 predictors and one separate target.
 * It refuses to edit the form when any response exists, preserving collected data.
 * The form is left closed after update; review and enable it manually if appropriate.
 */
const BD_FORM_ID = '18Vhtt4NHdhAkkrQe8zVMv3ESyny_-ulVrUauin2brPA';

const BD_OCCUPATIONS = [
  'Not Alive', 'Retired', 'Housewife', 'Business owner',
  'Government employee', 'Private-sector employee',
  'Self-employed / Daily-wage worker', 'Other', 'Unknown / Prefer not to say'
];
const BD_PARENT_EDUCATION = [
  'No formal education', 'Primary', 'Lower secondary', 'Secondary / SSC',
  'Higher secondary / HSC', "Bachelor's degree", 'Postgraduate degree',
  'Other', 'Unknown / Prefer not to say'
];
const BD_GENDER = ['Female', 'Male', 'Another identity', 'Unknown / Prefer not to say'];
const BD_MARITAL = [
  'Single / Never married', 'Married', 'Separated', 'Divorced', 'Widowed',
  'Other', 'Unknown / Prefer not to say'
];
const BD_YES_NO_UNKNOWN = ['Yes', 'No', 'Unknown / Prefer not to say'];
const BD_FINANCIAL = [
  'Poor (below ৳25,000)', 'Below Average (৳25,000–39,999)',
  'Average (৳40,000–69,999)', 'Higher Middle Class (৳70,000–149,999)',
  'Rich (৳150,000+)', 'Undisclosed'
];
const BD_EMPLOYMENT = [
  'Unemployed', 'Part-time', 'Full-time', 'Private Tuition',
  'Freelancing/Outsourcing', 'Own Business', 'Undisclosed'
];

// This order is shared with src/bangladesh_collection.py and schema v3.
const BD_QUESTIONS = [
  { title: 'HSC / Equivalent GPA', field: 'hsc_equivalent_gpa', type: 'number', min: 0, max: 5, help: 'Enter the reported GPA from 0 to 5, or leave blank if unknown.' },
  { title: 'Age at enrollment', field: 'age_at_enrollment', type: 'number', min: 15, max: 100, integer: true, help: 'Enter age in whole years.' },
  { title: "Mother's occupation", field: 'mother_occupation_bangladesh', type: 'list', choices: BD_OCCUPATIONS },
  { title: "Father's occupation", field: 'father_occupation_bangladesh', type: 'list', choices: BD_OCCUPATIONS },
  { title: "Mother's highest education", field: 'mother_education_bangladesh', type: 'list', choices: BD_PARENT_EDUCATION },
  { title: "Father's highest education", field: 'father_education_bangladesh', type: 'list', choices: BD_PARENT_EDUCATION },
  { title: 'Gender', field: 'gender_bangladesh', type: 'list', choices: BD_GENDER },
  { title: 'Marital status', field: 'marital_status_bangladesh', type: 'list', choices: BD_MARITAL },
  { title: 'Scholarship holder', field: 'scholarship_holder', type: 'list', choices: BD_YES_NO_UNKNOWN },
  { title: 'Have any debt', field: 'debtor', type: 'list', choices: BD_YES_NO_UNKNOWN },
  { title: 'Tuition fees up to date', field: 'tuition_fees_up_to_date', type: 'list', choices: BD_YES_NO_UNKNOWN },
  { title: 'Courses enrolled in the relevant semester', field: 'courses_enrolled', type: 'number', min: 0, max: 500, integer: true, help: 'Enter the number of courses, not credit hours.' },
  { title: 'Evaluations completed in the relevant semester', field: 'evaluations_completed', type: 'number', min: 0, max: 1000, integer: true, help: 'Enter a whole-number count.' },
  { title: 'Courses passed in the relevant semester', field: 'courses_passed', type: 'number', min: 0, max: 500, integer: true, help: 'Enter a whole-number count.' },
  { title: 'Semester GPA (native university scale)', field: 'semester_gpa_native', type: 'number', min: 0, max: 5, help: 'Enter the GPA on your university scale, from 0 to 5.' },
  { title: 'Last-semester attendance percentage', field: 'last_semester_attendance_percent', type: 'number', min: 0, max: 100, help: 'Enter 0 to 100, or leave blank if unknown.' },
  { title: 'Family financial condition (monthly household income bracket)', field: 'family_financial_condition', type: 'list', choices: BD_FINANCIAL, help: 'These provisional income brackets are not official poverty definitions.' },
  { title: 'Student employment', field: 'student_employment', type: 'list', choices: BD_EMPLOYMENT },
  { title: 'Current Academic Status', field: 'current_academic_status', type: 'list', required: true, choices: [
    'Running (Currently Studying)', 'Passed (Graduated)', 'Dropout (Discontinued Studies)'
  ] }
];

function updateExistingBangladeshResearchForm() {
  const form = FormApp.openById(BD_FORM_ID);
  const existingResponses = form.getResponses().length;
  if (existingResponses > 0) {
    throw new Error(
      'No changes made. This form already has ' + existingResponses +
      ' response(s). Export and safely archive the responses before deciding how to migrate the form.'
    );
  }

  // No form changes happen until after the response-preservation check above.
  form.setAcceptingResponses(false);
  form.setCollectEmail(false);

  // Empty existing form: remove old questions and section items, then build one section only.
  for (let i = form.getItems().length - 1; i >= 0; i--) {
    form.deleteItem(i);
  }
  form.setTitle('Bangladesh University Student Questionnaire');
  form.setDescription('Answer the 18 predictor questions and the required current academic status. Leave an optional predictor blank if it is unknown or unavailable. Do not enter names, email addresses, or official student IDs.');
  form.setLimitOneResponsePerUser(false);
  form.setShowLinkToRespondAgain(false);

  BD_QUESTIONS.forEach((question) => addBangladeshQuestion(form, question));

  const items = form.getItems();
  if (items.length !== 19 || items.some((item) => item.getType() === FormApp.ItemType.PAGE_BREAK || item.getType() === FormApp.ItemType.SECTION_HEADER)) {
    throw new Error('Form verification failed: expected exactly 19 questions and no additional sections.');
  }
  Logger.log('Updated existing form. Verified 18 predictors plus one required target question. Email collection is off; response collection remains off.');
}

function addBangladeshQuestion(form, question) {
  if (question.type === 'list') {
    let item = form.addListItem().setTitle(question.title).setChoiceValues(question.choices).setRequired(Boolean(question.required));
    if (question.help) item.setHelpText(question.help);
    return;
  }

  let validation = FormApp.createTextValidation()
    .requireNumberBetween(question.min, question.max)
    .setHelpText('Enter a number from ' + question.min + ' to ' + question.max + '. ' + (question.help || ''))
    .build();
  form.addTextItem()
    .setTitle(question.title)
    .setHelpText(question.help || 'Enter a number in the stated range, or leave blank if unknown.')
    .setValidation(validation)
    .setRequired(Boolean(question.required));
}
