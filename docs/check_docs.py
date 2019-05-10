import sys
sys.path.append("/home/localadmin1/projects/pyeo")
import pyeo.core as pyeo

def check_pyeo_docs(doc_path):
    pyeo_func_list = dir(pyeo)[20:]
    with open(doc_path, 'r') as index:
        print("Searching for missing functions")
        text = index.read()
        for func in pyeo_func_list:
            if text.find(func) == -1:
                print(func)
                
check_pyeo_docs(r"/home/localadmin1/projects/pyeo/docs/source/index.rst")
