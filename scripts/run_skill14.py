import sys
sys.path.insert(0, ".")
from zindian.skills.skill_14_inference import run

result = run(branch_name="anchor-baseline")
print(result)