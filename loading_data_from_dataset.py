import numpy as np
import csv
from tqdm import tqdm

class Student():
    def __init__(self, ID, score, assignments):
        self.ID = ID
        self.score = score
        self.assignments = assignments

class Assignment(): 
    def __init__(self, ID, problems):
        self.ID = ID
        self.problems = problems

class Problem(): 
    def __init__(self, ID, codeStates):
        self.ID = ID
        self.codeStates = codeStates

class CodeState():
    def __init__(self, ID, code, compileResult, compileMessageType, compileMessageData, score, time):
        self.ID = ID
        self.code = code
        self.compileResult = compileResult
        self.compileMessageType = compileMessageType
        self.compileMessageData = compileMessageData
        self.score = score
        self.time = time

def load_data():

    codeStates = {}
    with open('./Datasets/CodeStates.csv', mode='r', encoding="utf8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for line in csv_reader:
            codeStates[line[0]] = line[1] 
            
    mainTable = {}
    with open('./Datasets/MainTable.csv', mode='r', encoding="utf8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for line in csv_reader:
            # Student
            if line[0] not in mainTable:
                mainTable[line[0]] = {}
            # Assignment
            if line[5] not in mainTable[line[0]]:
                mainTable[line[0]][line[5]] = {}
            # Problem
            if line[6] not in mainTable[line[0]][line[5]]:
                mainTable[line[0]][line[5]][line[6]] = {}
            # CodeState
            if line[7] not in mainTable[line[0]][line[5]][line[6]]:
                # mainTable[Student][Assignment][Problem][Codestate]
                mainTable[line[0]][line[5]][line[6]][line[7]] = {} 
            mainTable[line[0]][line[5]][line[6]][line[7]]['time'] = line[2]
            mainTable[line[0]][line[5]][line[6]][line[7]]['code'] = codeStates[line[7]]
            if line[10] != '':
                mainTable[line[0]][line[5]][line[6]][line[7]]['score'] = line[10] 
            if line[11] != '':
                mainTable[line[0]][line[5]][line[6]][line[7]]['compileResult'] = line[11]
            if line[12] != '':
                mainTable[line[0]][line[5]][line[6]][line[7]]['compileMessageType'] = line[12]
                mainTable[line[0]][line[5]][line[6]][line[7]]['compileMessageData'] = line[13]
    
    studentScore = {}
    with open('./Datasets/Subject.csv', mode='r', encoding="utf8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for line in csv_reader:
            studentScore[line[0]] = line[1] 

    studentDataset = []
    for student in tqdm(mainTable.keys()):
        assignments = []
        for assignment in mainTable[student]:
            problems = []
            for problem in mainTable[student][assignment]:
                codeStates = []
                for codeState in mainTable[student][assignment][problem]:
                    compileMessageType = ''
                    compileMessageData = ''
                    if 'complieMessageType' in mainTable[student][assignment][problem][codeState]:
                        compileMessageType = mainTable[student][assignment][problem][codeState]['compileMessageType']
                        compileMessageData = mainTable[student][assignment][problem][codeState]['compileMessageData']
                    cS = CodeState(
                        codeState, 
                        mainTable[student][assignment][problem][codeState]['code'],
                        mainTable[student][assignment][problem][codeState]['compileResult'],
                        compileMessageType,
                        compileMessageData,
                        mainTable[student][assignment][problem][codeState]['score'],
                        mainTable[student][assignment][problem][codeState]['time']
                    )
                    codeStates.append(cS)
                pb = Problem(problem, codeStates)
                problems.append(pb)
            ass = Assignment(assignment, problems)
            assignments.append(ass)
        stu = Student(student, studentScore[student], assignments)
        studentDataset.append(stu)

    return np.array(studentDataset)