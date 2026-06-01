# Machine-Learning
> Based on CS 325/780 Syllabus (Summer 2026) | Prof. Paul R. Cesaretti
---


## Table of Contents

1. [What Even Is Machine Learning?](#chapter-0-what-even-is-machine-learning)
2. [The Formal Model](#chapter-1-the-formal-model--lets-get-precise)
3. [ERM - Empirical Risk Minimization](#chapter-2-erm--empirical-risk-minimization)
4. [PAC Learning](#chapter-3-pac-learning--probably-approximately-correct)
5. [VC Dimension](#chapter-4-vc-dimension--how-powerful-is-your-rule-class)
6. [Linear Models](#chapter-5-linear-models--the-simplest-rules)
7. [Loss Functions](#chapter-6-loss-functions--how-mad-should-we-be)
8. [Feature Transformations](#chapter-7-feature-transformations--cheating-with-math)
9. [Gradient Descent](#chapter-8-gradient-descent--how-computers-actually-learn)
10. [Perceptron Algorithm](#chapter-9-perceptron--the-oldest-learning-algorithm)
11. [Maximum Likelihood Estimation (MLE)](#chapter-10-maximum-likelihood-estimation-mle)
12. [Overfitting](#chapter-11-overfitting--the-silent-killer)
13. [ε-Representative Sample](#chapter-12-ε-representative-sample)
14. [Axis-Aligned Rectangles](#chapter-13-axis-aligned-rectangles)
15. [Online Learning](#chapter-14-online-learning)

---

## Chapter 0: What Even Is Machine Learning?

**The simplest definition ever:**

> Machine learning is when a computer learns from examples instead of following fixed rules.

**Example:**
- **Traditional programming:** You tell the computer, "If the email has the word 'lottery', mark it as spam."
- **Machine learning:** You show the computer 10,000 emails, some spam, some not. It figures out its own rules.

**The big idea:** We don't tell the computer *what* to learn. We tell it *how to learn*.

---

## Chapter 1: The Formal Model — Let's Get Precise

### The setup (imagine this)

You are a teacher. The computer is a student.

You have:

| Symbol | Meaning | Example |
|--------|---------|---------|
| **X** | Inputs (questions) | A picture of an animal |
| **Y** | Labels (answers) | "cat" or "dog" |
| **D** | Distribution (real world) | Some animals appear more often |
| **H** | Hypothesis class (all possible rules) | All possible spam filters |
| **Loss function** | How mad you get when wrong | 1 if wrong, 0 if correct |

### The two errors you MUST know

| Name | What it means | Formula |
|------|---------------|---------|
| **Training error** | How many answers wrong on homework | `(wrong answers) / (total homework questions)` |
| **Test error (generalization error)** | How many wrong on the real exam | Same formula, but on NEW questions |

**The nightmare:** Low training error but high test error = memorized homework, didn't learn.

---

## Chapter 2: ERM — Empirical Risk Minimization

**Risk** = fancy word for "error" or "loss".

**Empirical** = based on data we've seen.

**Empirical Risk Minimization** = Pick the rule that does best on the homework.

### Pseudocode (the simplest algorithm in ML)

```python
def ERM(homework_questions, homework_answers):
    best_rule = None
    best_score = infinity
    
    for each possible_rule in H:
        errors = 0
        for i in range(len(homework_questions)):
            if possible_rule(homework_questions[i]) != homework_answers[i]:
                errors += 1
        if errors < best_score:
            best_score = errors
            best_rule = possible_rule
    
    return best_rule
```

**Problem:** If H is huge (like all possible rules ever), this loop takes forever. That's why we need clever algorithms.

---

## Chapter 3: PAC Learning — Probably Approximately Correct

This sounds scary. It's actually beautiful.

**Question:** If I give you a sample of data, can you *guarantee* your rule will be good?

**Answer:** No. Because luck is involved. The sample might be weird.

**PAC says:** We can't guarantee 100%. But we can be *probably* (high chance) *approximately* (close enough) correct.

### The four parameters

| Symbol | Meaning | Example |
|--------|---------|---------|
| **ε (epsilon)** | How wrong we allow | 0.05 = 5% error max |
| **δ (delta)** | How often we fail | 0.01 = 99% confidence |
| **m** | Number of examples | 1000 emails |
| **\|H\|** | Number of possible rules | 1 million |

### The formula (for finite H, perfect data)

```
m ≥ (1/ε) × ln(|H|/δ)
```

**In English:** You need at least this many examples.

Let's use it :

```
|H| = 1000 rules
ε = 0.1 (10% error allowed)
δ = 0.05 (95% confidence)

m ≥ (1/0.1) × ln(1000/0.05)
m ≥ 10 × ln(20,000)
m ≥ 10 × 9.9 = 99 examples
```

**Why this matters:** Even with 1000 possible rules, you only need ~100 examples to learn well. That's amazing!

---

## Chapter 4: VC Dimension — How Powerful Is Your Rule Class?

> The problem : What if H is infinite? We can't use |H| in the formula.

**Example:** All lines in 2D. There are infinitely many lines.

#### Enter VC dimension (Vapnik-Chervonenkis)

**The intuition:** How many points can you "shatter" (separate in every possible way)?
```
 Imagine you have 3 cookies on a table. You have a ruler (a line). You can put the line anywhere.
 Can you separate the cookies in every possible labeling?
 Labeling = which cookies are "+" and which are "-".
 For 3 cookies: 2³ = 8 possible patterns.
 With a line in 2D: **YES!** You can get all 8 patterns (unless cookies are in a straight line).
 For 4 cookies: Can you get all 16 patterns? **NO.** Try the XOR pattern (diagonal opposites) — no single line works.
```
**Therefore:** VC dimension of lines in 2D = **3**.

### The VC formula (sample complexity)

```
m ≥ O( (VC + ln(1/δ)) / ε² )
```

**What this means:** If your rule class is more powerful (higher VC), you need more data.

---

## Chapter 5: Linear Models — The Simplest Rules

### What is a linear model?

A rule of the form:

```
prediction = weight1 × feature1 + weight2 × feature2 + ... + bias
```

**2D example:** `y = 2x + 1` is a line.

**Classification version:** If `w·x + b > 0`, predict +; else predict -.

### Why linear models are great

- Easy to understand
- Fast to compute
- Don't overfit much

### Why linear models are limited

**Cannot learn XOR.**

Try it: XOR says (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0. Draw these points. No single straight line separates them.

**Solution:** Feature transformations (Chapter 7).

---
## Chapter 6: Loss Functions — How Mad Should We Be?

A loss function answers: "The prediction was off by X. How bad is that?"

### For regression (predicting numbers)

**Squared loss (MSE):** `(predicted - actual)²`

Example: Predict 7, actual is 10. Loss = 9.

Why square? Makes big errors VERY bad.

### For classification (predicting categories)

**0-1 loss:** `0 if correct, 1 if wrong`

Simple but hard to optimize (not smooth).

**Log loss (cross-entropy) for logistic regression:**

```
If actual = 1:  loss = -log(predicted_probability)
If actual = 0:  loss = -log(1 - predicted_probability)
```

Example: Actual=1, predicted=0.9 → loss = -log(0.9) ≈ 0.105

Actual=1, predicted=0.1 → loss = -log(0.1) ≈ 2.30 (much worse)

---

## Chapter 7: Feature Transformations — Cheating With Math

### The problem

Your data isn't linearly separable. You need a curve.

### The trick

Don't change the algorithm. Change the data.

### Example: Circles problem

**Original 2D data:** Points inside a circle are +, outside are -.

No straight line works.

**Feature transform:** `φ(x,y) = x² + y²` (distance from origin squared)

Now in 1D:
- Inside circle: distance < R² → value < threshold
- Outside: distance > R² → value > threshold

**In 1D, a single threshold (a point on a line) separates them perfectly!**

### The golden rule

> A feature transform lets linear models solve non-linear problems by mapping data into a new space where the pattern becomes straight.

### Danger of feature transforms

- Too many features → overfitting
- You need to know the pattern in advance (or try many transforms)

---

## Chapter 8: Gradient Descent — How Computers Actually Learn

### The big idea

You want to find the best weights `w`. You have a loss function `L(w)`.

Imagine you're blindfolded on a mountain. You want to get to the bottom (minimum loss).

**Gradient descent:** Feel the ground. Which direction goes down? Take a small step that way. Repeat.

### The math simplified

```
w_new = w_old - learning_rate × gradient
```

- **Gradient** = direction of steepest ascent (so we go opposite)
- **Learning rate** = how big a step to take

### Three versions

| Version | What it does | Speed | Accuracy |
|---------|--------------|-------|----------|
| Batch GD | Uses ALL data for each step | Slow | Precise |
| SGD (Stochastic) | Uses 1 random example per step | Fast | Noisy |
| Mini-batch | Uses ~32 examples per step | Medium | Good |

### Pseudocode for SGD (Logistic Regression)

```python
w = [0, 0, 0, ...]  # initialize weights

for each training example (x, y):
    prediction = 1 / (1 + e^(-w·x))  # sigmoid
    error = prediction - y
    gradient = error × x
    w = w - learning_rate × gradient
```

---
## Chapter 9: Perceptron — The Oldest Learning Algorithm

### History

Invented in 1958. The first neural network. People got VERY excited. Then they realized it couldn't learn XOR. ("AI winter" happened.)

### How it works

```python
w = [0, 0, ...]

repeat:
    for each (x, y):
        if y * (w·x) ≤ 0:  # misclassified
            w = w + y × x   # update
until no mistakes
```

### The intuition

When you make a mistake, you move the decision boundary toward the misclassified point.

### Guarantee (Novikoff theorem)

If the data is linearly separable with margin γ, perceptron makes at most (R²/γ²) mistakes.

**Margin** = how "thick" the separator is. Bigger margin = fewer mistakes.

### What if data isn't separable?

Perceptron never stops. Solutions:
1. Run for a fixed number of passes, keep best weights
2. Use the "voted perceptron"
3. Use SVM instead

---

## Chapter 10: Maximum Likelihood Estimation (MLE)

### The question

You have a biased coin. You flip it 10 times. You get 7 heads.

What is the most likely probability of heads?

### Likelihood

```
Likelihood(probability = p) = p⁷ × (1-p)³
```

This is the probability of seeing exactly that sequence.

### Maximum likelihood

Find p that maximizes p⁷(1-p)³.

**Calculus answer:** p = 7/10 = 0.7

**Shocking insight:** The most likely probability is just the fraction of heads!

### Negative log-likelihood

Instead of maximizing `p⁷(1-p)³`, we minimize:

```
- [7×ln(p) + 3×ln(1-p)]
```

**Why do this?**

1. Products become sums (easier math)
2. Numbers don't get tiny (more stable)
3. The function becomes convex (easier to optimize)

---

## Chapter 11: Overfitting — The Silent Killer

### The story

Student memorizes all homework answers. Gets 100% on homework. Fails final exam because questions are different.

That's overfitting.

Why overfitting happens : The model is too powerful. It learns the noise, not the signal.

 ***Example***

10 points. Fit a 9th-degree polynomial: perfect through all points! But useless for prediction.

### How to prevent overfitting

| Method | How it works |
|--------|---------------|
| More data | Hard to overfit with 1M examples |
| Simpler model | Fewer parameters |
| Regularization | Penalize large weights |
| Cross-validation | Test on held-out data |
| Early stopping | Stop training before overfitting |

**The bias-complexity tradeoff**

```
Total error = Bias² + Variance + Noise
```

- **Bias** = How wrong is the best rule in H? (High if H is too simple)
- **Variance** = How much does the rule change with different data? (High if H is too complex)

**The sweet spot:** Medium complexity.

---

## Chapter 12: ε-Representative Sample

### The definition

A sample S is ε-representative if for every rule h in H:

```
|training_error(h) - true_error(h)| ≤ ε
```

**In English:** Your homework grade is within ε of your real exam grade for EVERY possible rule.

**Why this is powerful**

If your sample is ε/2-representative, then:

```
true_error(ERM_hypothesis) ≤ best_possible_error + ε
```

**Translation:** The rule that does best on homework will do almost as well as the best possible rule in the real world.

---

## Chapter 13: Axis-Aligned Rectangles

### The problem

Learn to classify points in 2D as "inside rectangle" vs "outside".

***The efficient algorithm*** (realizable case)

```python
left = min(x of positive points)
right = max(x of positive points)
bottom = min(y of positive points)
top = max(y of positive points)

return rectangle = [left, right] × [bottom, top]
```

### Running time

**O(m)** where m = number of training examples.

**Sample complexity** : You need about `(4/ε) × log(2/δ)` examples.

**Example:** ε=0.1, δ=0.05 → about 4/0.1 × ln(2/0.05) = 40 × ln(40) ≈ 40 × 3.7 = 148 examples.

---

## Chapter 14: Online Learning

**In offline learning:** Get all data, then train, then predict.

**In online learning:**

```python
for t = 1 to infinity:
    receive x_t
    predict ŷ_t
    receive true y_t
    update model based on error
```

### Example: Positive rays

Domain X = real numbers. Rule: predict + if x ≥ a, else -.

Online algorithm: Start a = -∞. When you see a negative point, never lower a. When you see a positive point, raise a to that point.

**Mistake bound** : You make at most 2 mistakes per new "threshold" needed.

---

## Quick Reference Card for the Exam

| Concept | Formula / Definition |
|---------|---------------------|
| PAC sample complexity (finite H) | `m ≥ (1/ε) ln(|H|/δ)` |
| VC-dim sample complexity | `m ≥ O((VC + ln(1/δ))/ε²)` |
| Linear regression loss | `(1/m) Σ (wx_i - y_i)²` |
| Logistic regression loss | `-[y log(ŷ) + (1-y) log(1-ŷ)]` |
| Perceptron update | `w ← w + yx` |
| Gradient descent | `w ← w - η ∇L(w)` |
| Overfitting | Low train error, high test error |
| ε-representative | `|L_S(h) - L_D(h)| ≤ ε ∀ h` |
| Growth function | Max distinct labelings on m points |
| VC dimension | Largest d such that some set of d points is shattered |

---
