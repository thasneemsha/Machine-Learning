# Part 1: Foundations of Machine Learning

Machine learning is the automated detection of meaningful patterns in data. We want to program computers so they can "learn" from experience.

## The Learning Setup

* **Input:** Training data (examples of past experiences)
* **Output:** A hypothesis/predictor that can perform a task on new data
* **Goal:** Convert experience into expertise

---

## 2. The Formal Model (Book Terminology)

| Symbol                        | Name                   | What It Means                                       |
| ----------------------------- | ---------------------- | --------------------------------------------------- |
| `X`                           | Domain set             | The set of all possible objects (e.g., all papayas) |
| `Y`                           | Label set              | Possible answers (e.g., {tasty, not tasty} = {0,1}) |
| `S = ((x₁,y₁), ..., (xₘ,yₘ))` | Training set           | The data we have (labeled examples)                 |
| `h : X → Y`                   | Hypothesis / Predictor | Our guess at the pattern                            |
| `H`                           | Hypothesis class       | The set of all hypotheses we're willing to consider |
| `D`                           | Distribution           | How the world generates examples (unknown!)         |
| `f : X → Y`                   | Target function        | The TRUE labeling rule (also unknown!)              |

>1. **D** = How nature picks examples (e.g., probability of seeing a green papaya)
>2. **f** = The correct labeling rule (e.g., green papayas are tasty)

---

## The Two Types of Error (Measure of Success)

### Error Type 1: True Error / Generalization Error `L_D(h)`


```text
    L_(D,f)(h) = P_{x~D}[h(x) ≠ f(x)] = D({x : h(x) ≠ f(x)})
```

In the agnostic setting (no perfect f):

```text
     L_D(h) = P_(x,y)~D [ h(x) ≠ y ]
```

The error of a model/hypothesis class is the probability that it mislabels an element sampled from the underlying distribution D.

**Translations** :  If you randomly pick a NEW papaya from the store, what's the chance your hypothesis guesses the wrong taste?

> But Even if we had an algorithm that computes h in H, we don't know the distribution D, or the labeling function F. We **cannot** compute `L_D(h)` directly.

---
###  Error Type 2: Empirical Error / Training Error `L_S(h)`

```text
L_S(h) = |{ i ∈ [m] : h(x_i) ≠ y_i }| / m
```
> Define a Training Error: Empirical Risk

```text
L_S(h) =  (# wrong answers)
          -----------------
          (total questions)
```

**Translations** : On the papayas you already cut open and tasted, how many did your hypothesis get wrong?

> unlike the previous error, we **can** compute it because we already have the data.

---

## Empirical Risk Minimization (ERM)


```text
ERM_H(S) ∈ argmin_{h ∈ H} L_S(h)
```

* Look at all hypotheses in `H`.
* Choose the one that makes the **fewest mistakes** on the training data.

We assume realizability:

```text
∃ h* ∈ H such that

L_(D,f)(h*) = 0
```

> This means there exists a perfect hypothesis in the class.
---

## When Does `L_S(h) = 0`?

### Case 1: Memorization Hypothesis (Overfitting)

```text
            y_i     if ∃ i such that x_i = x
h_S(x) =
            0       otherwise
```

This gets:

```text
Training Error = 0
```

* but may perform terribly on unseen data.
* It memorizes answers instead of learning patterns.

### Case 2: Realizable Setting

If there exists a hypothesis in `H` that labels every training example correctly, then ERM can achieve:

```text
L_S(h) = 0
```

* Just because you *can* get zero training error does **not** mean the model generalizes well.
* That is overfitting.

---

## Overfitting (The Memorization Trap)


Overfitting occurs when:

```text
Low empirical error
BUT
High true error
```

Suppose:

* True rule: papayas are tasty if orange AND soft.
* Training set: 10 orange-soft tasty papayas.
* Memorization rule: only those exact 10 papayas are tasty.

Result:

```text
L_S(h_S) = 0
```

Perfect on training data.

```text
L_D(h_S) = very high
```

Terrible on new papayas.

* It can memorize noise instead of learning patterns.
* Solution is to restrict `H`.
* Smaller hypothesis classes are harder to overfit.
