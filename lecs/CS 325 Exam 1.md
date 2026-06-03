
### Question 1: PAC & Agnostic PAC Learning

```
> Define PAC & Agnostic PAC.
> What is the difference?
> How does the definition of the generalization error change?
> How does the empirical risk definition change?
```

### a. PAC Learning (Probably Approximately Correct)

**Definition:**

 * A hypothesis class `H` is PAC-learnable if there exists a learning algorithm such that 
 * for every distribution `D`, every target concept `h* ∈ H`, every accuracy parameter `ε > 0`, and confidence parameter `δ > 0`, 
 * the algorithm returns a hypothesis `h` satisfying `L_D(h) ≤ ε` with probability at least `1 − δ`.

```text
P_{S~D^m}[L_(D,f)(h) ≤ ε] ≥ 1 - δ
```

 * The PAC model assumes the **Realizability Assumption**:

`∃ h* ∈ H such that L_D(h*) = 0`

 * This means the true labeling function is contained in the hypothesis class.

---

### b. Agnostic PAC Learning
>The realizability assumption is removed.

**Definition:** 

* A hypothesis class `H` is agnostic PAC learnable if there exists `m_H:(0,1)^2 → N`
* and a learning algorithm such that for every `ε, δ ∈ (0,1)`
*  and every distribution `D` over `X × Y` (no realizability assumption), with `m ≥ m_H(ε,δ)` examples:

```text
P_{S~D^m}[L_D(h) ≤ min_{h'∈H} L_D(h') + ε] ≥ 1 - δ
```

---

### c. The Differences

| Aspect                   | PAC                             | Agnostic PAC             |         |       |   |          |
| ------------------------ | ------------------------------- | ------------------------ | ------- | ----- | - | -------- |
| Realizability assumption | Required (exists perfect h ∈ H) | NOT required             |         |       |   |          |
| Comparison target        | Zero error (perfect prediction) | Best possible error in H |         |       |   |          |
| Sample complexity        | O(log)                          | H                        | `δ ÷ ε` | O(log) | H | `δ ÷ ε²` |
| When to use              | Clean data                      | Noisy data               |         |       |   |          |

---

### d. How the Generalization Error **Definition Changes**

|                      | PAC                                    | Agnostic PAC                 |
| -------------------- | -------------------------------------- | ---------------------------- |
| Generalization Error | `L_(D,f)(h) = P_{x~D}[h(x) ≠ f(x)]`    | `L_D(h)=P_(x,y)~D[h(x) ≠ y]` |
| Measures             | Error against true labeling function f | Error against noisy labels   |

In PAC, labels come from a deterministic `f`.

In agnostic PAC, labels can be random or noisy.

---

### e. How the Empirical Risk Definition Changes

**It does NOT change.**

In both models:

```text
L_S(h) = (1/m) Σ[i=1→m] ℓ(h,z_i) = |{i ∈ [m] : h(x_i) ≠ y_i}| / m
```

**Professor's note:**

```text
L_S(h) = (# wrong answers)
         -----------------
         n(total questions)
```

* PAC: assume some `h ∈ H` can achieve zero error.
* Agnostic PAC: no such assumption.

---

## Question 2: Effect of ε and Hypothesis Class on Sample Complexity
```
> How does decreasing the approximation error `ε` affect the sample complexity `m`?
> If you use a simpler hypothesis class, what happens to the sample complexity `m`?
```
**Part 1: Decreasing ε**

As `ε` decreases, sample complexity `m` increases.
> As `ε ↓` The denominator becomes smaller. Therefore: m ↑

### Finite Hypothesis Class (Agnostic)

```text
m ≈ 2log(2|H|/δ) / ε²
```

### Finite Hypothesis Class (Realizable)

```text
m ≈ log(|H|/δ) / ε
```

> Think of ε as the size of your target.
>
> If you want to hit a bullseye (small ε), you need way more darts (m).

**Part 2: Simpler Hypothesis Class**

A simpler hypothesis class decreases sample complexity.

However, the approximation error may increase.
> The sample complexity depends on `log|H|`. If `|H|` decreases : `log|H|` decreases. Therefore: `m` decreases.

### Formula

```text
m_simple ≈ 2log(2|H_simple|/δ) / ε²

vs

m_complex ≈ 2log(2|H_complex|/δ) / ε²
```

Since:

```text
|H_simple| < |H_complex|
```

### The simpler class requires fewer samples.

* Simple class → low estimation error, high approximation error.
* Complex class → high estimation error, low approximation error.

The goal is to balance both.

---

## Question 3: C++ Programs with 100 Lines of Code

```
> Let `H` be the class of all C++ programs with 100 lines of code.

> Is this class PAC-learnable?

> If so, explain why and give the sample complexity.
```
let `H = { all C++ programs with exactly 100 lines }`
>PAC learnability for finite classes follows from the Fundamental Theorem of PAC Learning.
**Is It PAC-Learnable?** --> **YES** According to definition of PAC, Finite Hypothesis Classes Are PAC-Learnable

* Each line is a finite sequence of characters.
* The alphabet is finite.
* Maximum number of characters is bounded.
* Therefore the number of possible programs is finite.


---

### Sample Complexity

For finite classes:

`m_H(ε,δ) ≤ (log|H| + log(1/δ)) / ε`

Therefore:

`m = O((log|H| + log(1/δ))/ε)`

where `|H|` is the finite number of possible programs.

---


















