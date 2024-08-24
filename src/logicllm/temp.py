import os
import nltk
from nltk.inference import Prover9

# 检查环境变量
print("PROVER9:", os.environ.get('PROVER9'))

# 检查 Prover9 是否可以找到
prover = Prover9()
print("Prover9 path:", prover._find_binary('prover9'))
