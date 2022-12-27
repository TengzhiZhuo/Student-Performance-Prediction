import javalang
import numpy as np
from path_extractor import extracting_path
from tqdm import tqdm

def code_to_AST(code):
    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree

def fetch_codeState(students):
    students_output = []
    students_label = []

    for student in tqdm(students):
        for assignment in student.assignments:
            questions = []
            questions_label = []
            for problem in assignment.problems:
                codes = []
                codes_label = []
                for codeState in problem.codeStates:
                    try:
                        parsed = code_to_AST(codeState.code)
                    except:
                        parsed = "error"
                    label = codeState.score
                    codes.append(parsed)
                    codes_label.append(label)
                hashing_table = {}
                AST_paths = [extracting_path(java_code, max_length=8, max_width=2, hash_path=True, hashing_table=hashing_table) for java_code in codes]
                questions.append(AST_paths)
                questions_label.append(codes_label)
        students_output.append(questions)
        students_label.append(questions_label)
    
    return students_output, np.array(students_label)