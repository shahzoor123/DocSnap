import requests

url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

body = {
	"input": """I have an image of a patient'\''s medical report that contains important textual information. Please extract and transcribe all the text from the image accurately, including medical terminology, patient details, dates, and any numerical values. Maintain the formatting as closely as possible, such as headings, sections, and bullet points. If any text is unclear or partially visible, note it accordingly. Ensure that all medical terms, measurements, and data are transcribed correctly. formate should be in one paragraphs mention important stuff

Input: Patient Information

Name: Faraz Ahmed
Date of Birth: January 15, 1990
Gender: Male
Patient ID: FA123456
Date of Examination: August 20, 2024

Referring Physician: Dr. Ayesha Malik

Chief Complaint: Fever, chills, headache, and body aches for the past three days.

History of Present Illness: Faraz Ahmed, a 34-year-old male, presented with a three-day history of intermittent high-grade fever accompanied by chills, severe headaches, and generalized body aches. He reports feeling extremely fatigued and has experienced sweating and malaise. Faraz mentions having traveled to a malaria-endemic region two weeks prior to symptom onset. No history of similar symptoms in the past.

Past Medical History:

Generally healthy with no chronic illnesses.
No known drug allergies.
Not on any regular medications.
Social History:

Non-smoker.
Occasional alcohol consumption.
Recent travel to [Region/Country] with exposure to mosquito bites.
Physical Examination:

Vital Signs:

Temperature: 39.5°C (103.1°F)
Blood Pressure: 125/80 mmHg
Pulse: 98 bpm
Respiratory Rate: 18 breaths per minute
Oxygen Saturation: 98% on room air
General Appearance: Patient appears febrile and slightly dehydrated.

HEENT:

Head: Normocephalic, atraumatic.
Eyes: Conjunctivae clear.
Throat: Slightly erythematous without exudates.
Cardiovascular: Regular rate and rhythm, no murmurs.

Respiratory: Clear to auscultation bilaterally, no wheezes or crackles.

Abdomen: Soft, non-tender, no hepatosplenomegaly.

Extremities: No joint swelling or rash.

Laboratory Investigations:

Complete Blood Count (CBC):

Hemoglobin: 11.2 g/dL (Low)
Hematocrit: 33%
White Blood Cells: 4,500 /µL (Neutrophils: 60%, Lymphocytes: 35%, Monocytes: 5%)
Platelets: 95,000 /µL (Thrombocytopenia)
Peripheral Blood Smear:

Presence of Plasmodium falciparum trophozoites observed.
Rapid Diagnostic Test (RDT) for Malaria:

Positive for Plasmodium falciparum.
Diagnosis:

Primary: Uncomplicated Plasmodium falciparum Malaria
Secondary: Thrombocytopenia, likely related to malaria infection
Treatment Plan:

Antimalarial Therapy:

Initiate Artemisinin-based Combination Therapy (ACT), such as Artemether-Lumefantrine, as per current WHO guidelines.
Supportive Care:

Ensure adequate hydration with oral rehydration solutions.
Administer antipyretics (e.g., acetaminophen) to manage fever and discomfort.
Monitoring:

Daily follow-up to monitor response to therapy and check for potential complications.
Repeat CBC in 48 hours to assess hematological parameters.
Patient Education:

Advise on the importance of completing the full course of antimalarial medication.
Educate about preventive measures to avoid mosquito bites, such as using insect repellent, wearing long-sleeved clothing, and sleeping under insecticide-treated bed nets, especially during future travels to endemic areas.
Prognosis: With prompt and appropriate treatment, the prognosis for uncomplicated malaria is excellent. Close monitoring will ensure any complications are promptly addressed.

Physician'\''s Signature:

Dr. Ayesha Malik, MD
[Medical License Number]
[Contact Information]


Output: 

Input: Patient Information

Name: Faraz Ahmed
Date of Birth: January 15, 1990
Gender: Male
Patient ID: FA123456
Date of Examination: August 20, 2024

Referring Physician: Dr. Ayesha Malik

Chief Complaint: Fever, chills, headache, and body aches for the past three days.

History of Present Illness: Faraz Ahmed, a 34-year-old male, presented with a three-day history of intermittent high-grade fever accompanied by chills, severe headaches, and generalized body aches. He reports feeling extremely fatigued and has experienced sweating and malaise. Faraz mentions having traveled to a malaria-endemic region two weeks prior to symptom onset. No history of similar symptoms in the past.

Past Medical History:

Generally healthy with no chronic illnesses.
No known drug allergies.
Not on any regular medications.
Social History:

Non-smoker.
Occasional alcohol consumption.
Recent travel to [Region/Country] with exposure to mosquito bites.
Physical Examination:

Vital Signs:

Temperature: 39.5°C (103.1°F)
Blood Pressure: 125/80 mmHg
Pulse: 98 bpm
Respiratory Rate: 18 breaths per minute
Oxygen Saturation: 98% on room air
General Appearance: Patient appears febrile and slightly dehydrated.

HEENT:

Head: Normocephalic, atraumatic.
Eyes: Conjunctivae clear.
Throat: Slightly erythematous without exudates.
Cardiovascular: Regular rate and rhythm, no murmurs.

Respiratory: Clear to auscultation bilaterally, no wheezes or crackles.

Abdomen: Soft, non-tender, no hepatosplenomegaly.

Extremities: No joint swelling or rash.

Laboratory Investigations:

Complete Blood Count (CBC):

Hemoglobin: 11.2 g/dL (Low)
Hematocrit: 33%
White Blood Cells: 4,500 /µL (Neutrophils: 60%, Lymphocytes: 35%, Monocytes: 5%)
Platelets: 95,000 /µL (Thrombocytopenia)
Peripheral Blood Smear:

Presence of Plasmodium falciparum trophozoites observed.
Rapid Diagnostic Test (RDT) for Malaria:

Positive for Plasmodium falciparum.
Diagnosis:

Primary: Uncomplicated Plasmodium falciparum Malaria
Secondary: Thrombocytopenia, likely related to malaria infection
Treatment Plan:

Antimalarial Therapy:

Initiate Artemisinin-based Combination Therapy (ACT), such as Artemether-Lumefantrine, as per current WHO guidelines.
Supportive Care:

Ensure adequate hydration with oral rehydration solutions.
Administer antipyretics (e.g., acetaminophen) to manage fever and discomfort.
Monitoring:

Daily follow-up to monitor response to therapy and check for potential complications.
Repeat CBC in 48 hours to assess hematological parameters.
Patient Education:

Advise on the importance of completing the full course of antimalarial medication.
Educate about preventive measures to avoid mosquito bites, such as using insect repellent, wearing long-sleeved clothing, and sleeping under insecticide-treated bed nets, especially during future travels to endemic areas.
Prognosis: With prompt and appropriate treatment, the prognosis for uncomplicated malaria is excellent. Close monitoring will ensure any complications are promptly addressed.

Physician'\''s Signature:

Dr. Ayesha Malik, MD
[Medical License Number]
[Contact Information]


Output:""",
	"parameters": {
		"decoding_method": "greedy",
		"max_new_tokens": 200,
		"repetition_penalty": 1
	},
	"model_id": "ibm/granite-13b-chat-v2",
	"project_id": "0870e270-9bed-4a63-8b97-978cf72002ee",
	"moderations": {
		"hap": {
			"input": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			},
			"output": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			}
		}
	}
}

headers = {
	"Accept": "application/json",
	"Content-Type": "application/json",
	"Authorization": "Bearer YOUR_ACCESS_TOKEN"
}

response = requests.post(
	url,
	headers=headers,
	json=body
)

if response.status_code != 200:
	raise Exception("Non-200 response: " + str(response.text))

data = response.json()