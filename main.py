import pandas as pd

data = {
    "username": ["student_1", "student_2", "student_3", "student_4", "student_5", "student_6", "student_7", "student_8", "student_9", "student_10"],
    "age": [17, 18, 17, 17, 17, 18, 18, 17, 18, 17],
    "grade": [12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    "favorite_subjects": ["Math;Computer Science", "Biology;Chemistry", "Biology;Chemistry", "History;English", "Math;Biology", "Physics;Math", "Biology;English", "Chemistry;Biology", "Computer Science;Math", "Biology;Physics"],
    "hobbies": ["Reading;Gaming", "Reading;Drawing", "Writing;Reading", "Writing;Reading;Art", "Reading;Swimming", "Reading;Programming", "Drawing;Volunteering", "Reading;Gardening", "Programming;Gaming", "Reading;Drawing"],
    "preferred_work_environment": ["Indoor", "Indoor", "Indoor", "Indoor", "Indoor", "Indoor", "Indoor", "Indoor", "Indoor", "Indoor"],
    "strengths": ["Logical thinking;Programming", "Attention to detail;Analytical skills", "Empathy;Critical thinking", "Creativity;Writing", "Analytical skills;Problem-solving", "Logical thinking;Problem-solving", "Empathy;Communication", "Analytical skills;Attention to detail", "Logical thinking;Technical skills", "Empathy;Attention to detail"],
    "weaknesses": ["Public speaking", "Time management", "Public speaking", "Math", "Public speaking", "Communication", "Math", "Public speaking", "Public speaking", "Math"],
    "aptitude_test_scores": ["85;90;80", "88;92;85", "90;85;88", "72;78;80", "80;85;82", "88;92;89", "85;80;90", "82;85;87", "90;88;85", "87;85;90"],
    "career_interests": ["BCA", "Pharmacy", "MBBS", "Humanities", "Pharmacy", "BCA", "MBBS", "Pharmacy", "BCA", "MBBS"]
}

df = pd.DataFrame(data)
df.to_csv('student_career_data.csv', index=False)


