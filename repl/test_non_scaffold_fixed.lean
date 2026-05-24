import Mathlib
import Aesop

open BigOperators Real Nat Rat Finset Topology

theorem aime_1983_p1
  (x y z w : ℕ)
  (ht : 1 < x ∧ 1 < y ∧ 1 < z)
  (hw : 0 < w)
  (h0 : Real.logb x w = 24)
  (h1 : Real.logb y w = 40)
  (h2 : Real.logb (x * y * z) w = 12) :
  Real.logb z w = 60 := by

  have h_base_x : logb ↑w ↑x = 1 / 24 := by
    have h_base_x : 0 < (x : ℝ) ∧ (x : ℝ) ≠ 1 := by
      have hx : 1 < (x : ℝ) := by exact_mod_cast ht.1
      constructor
      · linarith
      · linarith
    have h_val_w : 0 < (w : ℝ) ∧ (w : ℝ) ≠ 1 := by
      constructor
      · norm_cast
      · intro hw1
        have h_log_w : logb ↑x ↑w = logb ↑x 1 := by rw [hw1]
        rw [h_log_w, logb_one] at h0
        norm_num at h0
    have h_reciprocal : logb ↑w ↑x = 1 / logb ↑x ↑w := by
      have h_nonzero : logb ↑x ↑w ≠ 0 := by
        rw [h0]
        norm_num
      have h_inv : logb ↑w ↑x * logb ↑x ↑w = 1 := by
        have h_inv : logb ↑w ↑x = 1 / logb ↑x ↑w := by
          have hx_pos : 0 < (x : ℝ) := by exact_mod_cast h_base_x.1
          have hx_ne_one : (x : ℝ) ≠ 1 := by exact_mod_cast h_base_x.2
          have hw_pos : 0 < (w : ℝ) := by exact_mod_cast h_val_w.1
          have hw_ne_one : (w : ℝ) ≠ 1 := by exact_mod_cast h_val_w.2
          rw [logb, logb, one_div, inv_div]
        have h_mul : (1 / logb ↑x ↑w) * logb ↑x ↑w = 1 := by
          apply one_div_mul_cancel h_nonzero
        rw [h_inv]
        exact h_mul
      field_simp [h_nonzero]
      linear_combination h_inv
    rw [h0] at h_reciprocal
    exact h_reciprocal
  have h_base_y : logb ↑w ↑y = 1 / 40 := by
    have h_inv : logb ↑w ↑y = 1 / logb ↑y ↑w := by
      have hy_pos : 0 < (y : ℝ) := by
        have : 1 < y := ht.2.1
        norm_cast
        exact Nat.zero_lt_of_lt this
      have hy_ne_one : (y : ℝ) ≠ 1 := by
        have : 1 < (y : ℝ) := Nat.one_lt_cast.mpr ht.2.1
        linarith
      have hw_pos : 0 < (w : ℝ) := by
        exact_mod_cast hw
      have hw_ne_one : (w : ℝ) ≠ 1 := by
        intro h_w_one
        -- If w = 1, then logb b w = 0 for any base b.
        -- Use h0: logb x w = 24.
        have h_w_one_f : (w : ℝ) = 1 := h_w_one
        rw [h_w_one_f, logb_one] at h0
        -- 0 = 24 is a contradiction.
        norm_num at h0
    
      have h_base_change_lhs : logb (w : ℝ) (y : ℝ) = Real.log y / Real.log w := by
        rw [logb]
      have h_base_change_rhs : logb (y : ℝ) (w : ℝ) = Real.log w / Real.log y := by
        rw [logb]
    
      rw [h_base_change_lhs, h_base_change_rhs]
      field_simp [log_ne_zero_of_pos_of_ne_one hy_pos hy_ne_one, log_ne_zero_of_pos_of_ne_one hw_pos hw_ne_one]
    rw [h_inv, h1]
  have h_base_xyz : logb ↑w (↑x * ↑y * ↑z) = 1 / 12 := by
    have h_recip : logb (↑w) (↑x * ↑y * ↑z) = 1 / logb (↑x * ↑y * ↑z) ↑w := by
      have h_prod_gt_one : 1 < (↑x * ↑y * ↑z : ℝ) := by
        norm_cast
        have hx : 1 < x := ht.1
        have hy : 1 < y := ht.2.1
        have hz : 1 < z := ht.2.2
        have hxy : 1 < x * y := one_lt_mul'' hx hy
        exact one_lt_mul'' hxy hz
      have h_w_pos : 0 < (↑w : ℝ) := by
        exact Nat.cast_pos.mpr hw
      have h_w_not_one : (↑w : ℝ) ≠ 1 := by
        intro h_w_one
        have h_w_val : (↑w : ℝ) = 1 := by
          rw [h_w_one]
        rw [h_w_val, logb, Real.log_one, zero_div] at h0
        norm_num at h0
      have h_log_inv : ∀ (a b : ℝ), 0 < a → a ≠ 1 → 0 < b → b ≠ 1 → logb a b = 1 / logb b a := by
        intro a b ha_pos ha_one hb_pos hb_one
        rw [Real.logb, Real.logb]
        field_simp [Real.log_ne_zero_of_pos_of_ne_one ha_pos ha_one, Real.log_ne_zero_of_pos_of_ne_one hb_pos hb_one]
      exact h_log_inv ↑w (↑x * ↑y * ↑z) h_w_pos h_w_not_one (lt_trans zero_lt_one h_prod_gt_one) (ne_of_gt h_prod_gt_one)
    rw [h_recip, h2]
  have h_log_sum : logb ↑w (↑x * ↑y * ↑z) = logb ↑w ↑x + logb ↑w ↑y + logb ↑w ↑z := by
    have h_prod1 : logb (↑w) (↑x * ↑y * ↑z) = logb (↑w) (↑x * ↑y) + logb (↑w) ↑z := by
      have h_prod_rules : ∀ (a b : ℝ), 0 < a → 0 < b → logb ↑w (a * b) = logb ↑w a + logb ↑w b := by
        intro a b ha hb
        have h_log_mul_identity : ∀ (base u v : ℝ), 0 < u → 0 < v → logb base (u * v) = logb base u + logb base v := by
          intro base u v hu hv
          unfold logb
          rw [log_mul (ne_of_gt hu) (ne_of_gt hv), add_div]
        exact h_log_mul_identity (↑w) a b ha hb
      have h_pos_xy : 0 < (↑x * ↑y : ℝ) := by
        have hx : 0 < x := Nat.lt_trans zero_lt_one ht.1
        have hy : 0 < y := Nat.lt_trans zero_lt_one ht.2.1
        have hxy : 0 < x * y := Nat.mul_pos hx hy
        norm_cast
      have h_pos_z : 0 < (↑z : ℝ) := by
        apply Nat.cast_pos.mpr
        exact Nat.lt_trans Nat.zero_lt_one ht.2.2
      have h_assoc : (↑x * ↑y * ↑z : ℝ) = (↑x * ↑y) * ↑z := by
        rfl
      rw [h_assoc]
      exact h_prod_rules (↑x * ↑y) ↑z h_pos_xy h_pos_z
    have h_prod2 : logb (↑w) (↑x * ↑y) = logb (↑w) ↑x + logb (↑w) ↑y := by
      have hx : (0 : ℝ) < x := by norm_cast; linarith [ht.1]
      have hy : (0 : ℝ) < y := by norm_cast; linarith [ht.2.1]
      rw [logb_mul]
      · linarith [hx]
      · linarith [hy]
    rw [h_prod1, h_prod2, add_assoc]
  have h_base_z : logb ↑w ↑z = 1 / 60 := by
    rw [h_base_xyz, h_base_x, h_base_y] at h_log_sum
    linarith
  have h_final : logb ↑z ↑w = 1 / (logb ↑w ↑z) := by
    -- Use change of base formula: logb b a = log a / log b
    rw [logb, logb]
    -- Ensure denominators are non-zero for field_simp
    have hz_gt_1 : 1 < (z : ℝ) := by norm_cast; exact ht.2.2
    have hz_pos : 0 < (z : ℝ) := by linarith
    have hz_ne_1 : (z : ℝ) ≠ 1 := by linarith
    have log_z_nz : log (z : ℝ) ≠ 0 := log_ne_zero_of_pos_of_ne_one hz_pos hz_ne_1
    
    have hw_gt_0 : 0 < (w : ℝ) := by norm_cast
    have hw_ne_1 : (w : ℝ) ≠ 1 := by
      intro h
      have h_log_w : log (w : ℝ) = 0 := by rw [h, log_one]
      have h0_expand : logb (x : ℝ) (w : ℝ) = log (w : ℝ) / log (x : ℝ) := rfl
      rw [h_log_w, zero_div] at h0_expand
      rw [h0_expand] at h0
      linarith
    have log_w_nz : log (w : ℝ) ≠ 0 := log_ne_zero_of_pos_of_ne_one hw_gt_0 hw_ne_1
    
    field_simp
  rw [h_final, h_base_z]
  norm_num
