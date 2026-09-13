/-
  qicert machine-checked core claims (Lean 4, Batteries only — no Mathlib).

  Scope (deliberately narrow — see HANDOFF §9b):
  * (A) Product-chaining of per-layer bounds: IF every layer's true
        quantity `op` is bounded by its certified value `L`, THEN the
        end-to-end product bound is sound. Formalized over ℕ (Lean core):
        the checked content is the COMPOSITION shape — pointwise bounds
        lifting through products — which is exactly the step the Python
        chaining code performs. The deployment reading over ℝ uses the
        same shape with the standard ordered-field lemmas (cited in the
        report, Eq. 1); each layer additionally passes an empirical
        soundness check `bound >= tight norm` in every recorded run.
        The single-layer analytic fact ‖AB‖ ≤ ‖A‖·‖B‖ stays an explicit
        assumption, not a claim.
  * (B) Guard soundness/completeness: the deployed checker's accept
        condition is proved sound (accepts nothing out-of-ball), complete
        (rejects nothing in-ball and in-range), and vacuous-margin-safe.
        This mirrors the Z3 falsification in tests/test_certify_guard.py,
        now as proof terms instead of solver queries.

  What is NOT claimed: any bound on a specific weight tensor (that lives
  in results/*/run.json), closed-loop stability, or hardware behavior.
-/

import Batteries

/-- Product of bounds, by explicit recursion (core-only). -/
def boundProd : List Nat → Nat
  | [] => 1
  | x :: xs => x * boundProd xs

/-- (A) Product chaining: pointwise bounds lift to the product bound. -/
theorem prod_le_prod_of_pointwise :
    ∀ (ops bounds : List Nat),
      ops.length = bounds.length →
      (∀ i : Nat, ∀ h1 : i < ops.length, ∀ h2 : i < bounds.length,
        ops[i] ≤ bounds[i]) →
      boundProd ops ≤ boundProd bounds := by
  intro ops
  induction ops with
  | nil =>
      intro bounds hlen _
      have hnil : bounds = [] := List.length_eq_zero_iff.mp (by simpa using hlen.symm)
      simp [hnil, boundProd]
  | cons o rest ih =>
      intro bounds hlen hall
      match bounds with
      | [] => simp at hlen
      | b :: brest =>
          simp only [boundProd]
          have hlen' : rest.length = brest.length := by simpa using hlen
          have hrest : ∀ i : Nat, ∀ h1 : i < rest.length,
              ∀ h2 : i < brest.length, rest[i] ≤ brest[i] := by
            intro i h1 h2
            have h1' : i + 1 < (o :: rest).length := by simpa using Nat.succ_lt_succ h1
            have h2' : i + 1 < (b :: brest).length := by simpa using Nat.succ_lt_succ h2
            have h := hall (i + 1) h1' h2'
            simpa using h
          have ih' := ih brest hlen' hrest
          have hob : o ≤ b := hall 0 (by simp) (by simp)
          exact Nat.mul_le_mul hob ih'

/-- (B) The deployed guard's accept condition, exactly as implemented in
    `python/qicert/certify/guard.py::action_in_certified_set`. -/
def guardAccept (action reference : Int) (margin : Int) (nActions : Nat) : Bool :=
  decide (action - reference ≤ margin ∧ margin ≥ 0 ∧
    0 ≤ action ∧ action < (nActions : Int))

/-- Soundness: an accepted action is always in-ball (and the margin sane). -/
theorem guard_sound (a r m : Int) (n : Nat)
    (h : guardAccept a r m n = true) :
    a - r ≤ m ∧ 0 ≤ m := by
  unfold guardAccept at h
  simp at h
  exact ⟨h.1, h.2.1⟩

/-- Completeness: an in-ball, in-range action with sane margin is accepted. -/
theorem guard_complete (a r m : Int) (n : Nat)
    (hball : a - r ≤ m) (hm : 0 ≤ m)
    (hlo : 0 ≤ a) (hhi : a < (n : Int)) :
    guardAccept a r m n = true := by
  unfold guardAccept
  simp [hball, hm, hlo, hhi]

/-- Vacuous margin: a negative margin accepts nothing. -/
theorem guard_negative_margin_rejects_all (a r m : Int) (n : Nat)
    (hm : m < 0) : guardAccept a r m n = false := by
  unfold guardAccept
  simp
  omega
