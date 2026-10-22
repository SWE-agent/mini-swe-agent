import sys
sys.path.insert(0, '.')
from agent import ToyAgent

agent = ToyAgent()
result = agent.run("what is 2+2")
print("Result:", result)