from prover.lean.verifier import verify_lean4_file, Lean4ServerScheduler
from pprint import pprint


code = open('problems/aime_1984_p5.lean').read()
# 1 run single request verify_lean4_file
result = verify_lean4_file(code, timeout=300)
print("1 run single request verify_lean4_file")
pprint(result)

# 2 run multiple requests with Lean4ServerScheduler
verifier = Lean4ServerScheduler(max_concurrent_requests=1, name='auto_sorrifier')
outputs_list = verifier.submit_all_request([dict(code=code), dict(code=code)])
outputs_list = verifier.get_all_request_outputs(outputs_list)
verifier.close()
print("2 run multiple requests with Lean4ServerScheduler")
pprint(outputs_list)