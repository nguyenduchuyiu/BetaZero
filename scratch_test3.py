import json
from gammazero.env.lean_env import LeanEnv
from gammazero.env.lean_verifier import Lean4ServerScheduler
from gammazero.search.sorrifier.sorrifier import Sorrifier

def main():
    candidate_code = """
set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_543 : (∑ k ∈ Nat.divisors (30 ^ 4), 1) - 2 = 123 := by
  
    -- Use the property that sum of 1s over a Finset is the cardinality of the set
    rw [Finset.sum_const, nsmul_eq_mul, mul_one]
    -- (Nat.divisors (30 ^ 4)).card is the number of divisors of 30^4
    -- 30^4 = (2 * 3 * 5)^4 = 2^4 * 3^4 * 5^4
    -- The number of divisors is (4+1) * (4+1) * (4+1) = 125
    have h : (Nat.divisors (30 ^ 4)).card = 125 := by
      -- We can use the prime factorization formula for divisor count
      have h_fact : 30 ^ 4 = 2 ^ 4 * 3 ^ 4 * 5 ^ 4 := by norm_num
      rw [h_fact]
      -- Using native divisor count computation or norm_num
      norm_num
    rw [h]
    norm_num
  
  """

    scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=60, name="manual_run_exact")
    try:
        lean = LeanEnv(scheduler)
        sorrifier = Sorrifier(scheduler, max_cycles=50)

        print("=== INPUT CODE ===")
        print(candidate_code)
        print("==================\n")

        patched = sorrifier.fix_code(candidate_code)

        print("=== PATCHED CODE ===")
        print(patched)
        print("====================\n")

    finally:
        scheduler.close()

if __name__ == "__main__":
    main()
