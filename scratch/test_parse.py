import re

code = """have h_cont : ContinuousOn (fun (x : ℝ) => x * sin x) (Set.Ioo 0 Real.pi) := by
  refine Continuous.mul ?_ ?_ |>.continuousOn
  · exact continuous_id
  · exact Real.continuous_sin
have h_val_at_pi_six : (Real.pi / 6) * sin (Real.pi / 6) < 2/3 := by
  calc
    (Real.pi / 6) * sin (Real.pi / 6) = (Real.pi / 6) * (1/2) := by
      rw [Real.sin_pi_div_six]
    _ = Real.pi / 12 := by ring
    _ < 2/3 := by
      sorry
have h_val_at_pi_two : (2/3 : ℝ) < (Real.pi / 2) * sin (Real.pi / 2) := by
  calc
    (2/3 : ℝ) < Real.pi / 2 := by
      sorry
    _ = (Real.pi / 2) * 1 := by ring
    _ = (Real.pi / 2) * sin (Real.pi / 2) := by rw [Real.sin_pi_div_two]
have h_IVT : ∃ x ∈ Set.Ioo 0 Real.pi, x * sin x = 2/3 := by
  have hx1 : Real.pi / 6 ∈ Set.Ioo 0 Real.pi := by
    constructor
    · positivity
    · nlinarith [Real.pi_pos]
  have hx2 : Real.pi / 2 ∈ Set.Ioo 0 Real.pi := by
    constructor
    · positivity
    · nlinarith [Real.pi_pos]
  have h_val_neg : (fun (x : ℝ) => x * sin x - 2/3) (Real.pi / 6) < 0 := by
    linarith
  have h_val_pos : 0 < (fun (x : ℝ) => x * sin x - 2/3) (Real.pi / 2) := by
    linarith
  have h_cont_g : ContinuousOn (fun (x : ℝ) => x * sin x - 2/3) (Set.Ioo 0 Real.pi) := by
    sorry

  sorry
exact h_IVT"""

def extract_sorry_vars(code: str) -> set[str]:
    lines = code.splitlines()
    sorry_vars = set()
    stack = []
    
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        
        while stack and indent <= stack[-1][0]:
            stack.pop()
            
        match = re.match(r"(?:have|let)\s+([a-zA-Z0-9_]+)\s*[:=]", stripped)
        if match:
            var_name = match.group(1)
            stack.append((indent, var_name))
            
        # Use regex to match exactly the word sorry, not e.g. sorry_but
        if re.search(r"\bsorry\b", stripped):
            if stack:
                sorry_vars.add(stack[-1][1])
                
    return sorry_vars

print(extract_sorry_vars(code))
