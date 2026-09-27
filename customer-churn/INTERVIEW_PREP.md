# Interview Preparation: Customer Churn & Retention Analytics System

Every number in this file comes from actually running `python -m src.train` (seed 42). Memorise the numbers in **bold**, because they are the ones interviewers use to check whether you really built the project.

---

## 0. Read this first (the honest part)

1. **Run everything yourself tonight.** Clone it, `pip install -r requirements.txt`, then run `src.data_loader`, `src.eda`, `src.train` and `src.predict`. Open every file and read every line. If you can't explain a line, you don't own it yet. Interviewers can tell within 3 questions whether you wrote something or downloaded it.
2. **Your DNN does NOT beat Logistic Regression** (AUC 0.845 vs 0.842). Don't hide it. Saying *"I built a baseline and the DNN only matched it, which is expected on small tabular data"* is a **strong** answer. Claiming the DNN is "better" is a weak answer, and a good interviewer will catch it.
3. **Accuracy is 0.74, which is barely above the 0.735 you'd get by always predicting "No churn".** If you lead with "74% accuracy" you'll get destroyed. Lead with **recall 0.80 and ROC-AUC 0.845**, and explain why accuracy is low (class weights, on purpose).
4. The resume-defense list says **"Extensive EDA"**, but your resume bullet says "Performed EDA". Keep it as "Performed EDA". 8 plot groups is solid, not "extensive". Don't inflate it.
5. "Identified correlations" is only literally true for the **numerical** columns (Pearson heatmap). For categorical columns (contract, payment, services) you compared **churn rates per group**, which is not correlation. Use the right words.
6. If someone asks "did you use any help/tutorials/AI?", don't lie. Say: *"I followed a structured approach and used references, but I ran every step and I can explain every decision."* That's only true if you do step 1.

---

## 1. The 2-minute explanation (memorise this)

> "My project is a Customer Churn & Retention Analytics System. Churn means a customer cancelling their service. For a telecom company, keeping a customer is cheaper than getting a new one, so the goal is to predict who is likely to leave so the retention team can act first.
>
> I used the IBM Telco Customer Churn dataset: 7,043 customers, 19 features like tenure, contract type, internet service, payment method and monthly charges. The target is Churn, Yes or No. It's imbalanced: about 26.5% churn.
>
> First I did EDA with Pandas, Seaborn and Matplotlib. The strongest patterns were contract type: month-to-month customers churn at about 43% vs under 3% for two-year contracts. Tenure: churners have a median tenure of 10 months vs 38 for customers who stay, and almost half of first-year customers churn. Price: churners pay higher monthly charges, especially fiber-optic customers without tech support or online security. And electronic-check users churn at 45%.
>
> For preprocessing I found that TotalCharges was stored as text because 11 new customers had blank values. I converted it to numeric and handled the missing values inside a Scikit-learn Pipeline with a ColumnTransformer: median imputation and standard scaling for numerical columns, one-hot encoding for categorical ones. I split into train, validation and test with stratification, and fitted the preprocessing only on training data to avoid data leakage.
>
> The model is a small TensorFlow/Keras neural network: two hidden Dense layers with 32 and 16 ReLU neurons, dropout of 0.3 after each, and a single sigmoid output that gives the churn probability. I trained it with binary cross-entropy and the Adam optimizer. For class imbalance I used balanced class weights, so missing a churner costs about 2.8 times more. Early stopping on validation loss stopped training at epoch 24 and restored the best weights from epoch 14.
>
> On the held-out test set it got a recall of about 0.80, meaning it catches 80% of churners, with ROC-AUC 0.845 and F1 0.62. Accuracy is 74%, which I don't rely on, because always predicting 'no churn' already gives 73.5%. I also trained a Logistic Regression baseline, and it performed almost the same, which taught me that on small tabular data a deep network isn't automatically better. Finally, I wrote a prediction function that takes a raw customer record, runs it through the saved preprocessor and model, and returns something like 'this customer has a 93% probability of churn'."

---

## 2. Key numbers (you must know these without looking)

| Item | Value |
|---|---|
| Dataset | IBM Telco Customer Churn, **7,043 rows × 21 columns** |
| Class balance | **5,174 No (73.5%) / 1,869 Yes (26.5%)** |
| Hidden missing values | **11 blank `TotalCharges`, all with tenure = 0** |
| Duplicates | 0 exact. 22 if you ignore customerID (kept, they're different customers) |
| Split | **Train 4,507 / Val 1,127 / Test 1,409** (64/16/20, stratified) |
| Features after encoding | **46** (3 numeric + 43 one-hot) |
| Class weights | **{0: 0.68, 1: 1.88}** |
| Architecture | **46 → 32 ReLU → Drop 0.3 → 16 ReLU → Drop 0.3 → 1 Sigmoid** |
| Parameters | **2,049** (46×32+32 = 1,504; 32×16+16 = 528; 16×1+1 = 17) |
| Training | Adam lr **0.001**, batch **32** (141 steps/epoch), max 100 epochs, EarlyStopping patience **10** |
| Stopped | Epoch **24**, best (restored) epoch **14** |
| Test DNN | Acc **0.740**, Precision **0.507**, Recall **0.799**, F1 **0.620**, ROC-AUC **0.845** |
| Confusion matrix | **TN 744, FP 291, FN 75, TP 299** |
| LR baseline | Acc 0.741, P 0.508, R 0.789, F1 0.618, AUC **0.842** |
| "Always No" | Accuracy **0.735**, recall 0 |
| DNN *without* class weights | Acc **0.805**, P 0.668, R **0.527**, F1 0.589, AUC 0.841 (TN 937, FP 98, FN 177, TP 197) |
| EDA facts | Month-to-month **42.7%**, 1-yr 11.3%, 2-yr **2.8%** · Electronic check **45.3%** · Fiber **41.9%** · Median tenure **10 vs 38** · First-year churn **47.4%** · Corr tenure–Churn **−0.35**, Monthly–Churn **+0.19**, tenure–Total **0.83** |

---

## 3. Question bank by category

Format for every question:
- **Say:** the short verbal answer
- **If they dig:** the deeper explanation
- **In my project:** the concrete example

### A. Project overview

**A1. Tell me about your project.**
- **Say:** The 2-minute pitch above. For a 30-second version: "I predicted telecom customer churn with a Keras neural network. I used Scikit-learn for the preprocessing pipeline and class weights, and Seaborn for EDA. It catches 80% of churners with ROC-AUC 0.845."
- **If they dig:** Walk through the pipeline in order: load → clean → split → fit preprocessor on train → class weights → train with early stopping → evaluate on test → save → predict.
- **In my project:** `src/train.py` runs this whole flow in one script.

**A2. What problem does it solve for the business?**
- **Say:** It ranks customers by churn risk so a retention team with a limited budget contacts the right people before they leave.
- **If they dig:** The value comes from *acting* on the prediction. A prediction nobody acts on is worth nothing. The business needs to choose how many customers to contact and what offer to make.
- **In my project:** 590 customers flagged in the test set, 299 of whom really churned. The team would focus on those.

**A3. What is churn? What exactly is your target?**
- **Say:** Churn means the customer cancelled. In this dataset, `Churn = Yes` means they left within the last month. I mapped Yes→1, No→0.
- **If they dig:** It's a binary classification problem. The model outputs P(Churn = 1 | customer features).
- **In my project:** `df["Churn"].map({"Yes": 1, "No": 0})` in `data_loader.py`.

**A4. Why this dataset?**
- **Say:** It's a standard public churn dataset with realistic telecom features (contract, services, billing) and a real class imbalance, so it covers all the practical problems: dirty column, categorical data, imbalance.
- **If they dig:** Its limitation is that it's a single snapshot with no usage history or time series, and it's a sample dataset from IBM, not live company data.
- **In my project:** 7,043 rows, 19 usable features.

**A5. What was the hardest part?**
- **Say:** Choosing the right metric. Accuracy looked fine but was meaningless, because always predicting "no churn" gives 73.5%.
- **If they dig:** Class weights *lowered* accuracy but raised recall a lot. I had to justify why a lower accuracy is the better model for this business goal.
- **In my project:** Accuracy 0.740 vs baseline 0.735, but recall 0.799 vs 0.

**A6. What did you learn?**
- **Say:** Three things. Data leakage matters: fit only on train. Accuracy lies on imbalanced data. And a deep network isn't automatically better than Logistic Regression on tabular data.
- **In my project:** DNN AUC 0.845 vs LR 0.842.

### B. Python (based on this project)

**B1. How is your code organised, and why modules instead of one notebook?**
- **Say:** Each file does one job: `data_loader`, `preprocessing`, `model`, `train`, `evaluate`, `predict`, `eda`. The notebook imports from them, so there's no duplicated logic.
- **If they dig:** Modules are reusable, testable and importable. `predict.py` can be used by an app without running training.
- **In my project:** `notebooks/churn_eda.ipynb` calls `src.eda` functions.

**B2. What does `if __name__ == "__main__":` do?**
- **Say:** The code under it only runs when the file is executed directly, not when it's imported.
- **In my project:** `predict.py` runs two demo customers when executed, but when the notebook imports `predict_churn`, the demo doesn't run.

**B3. Why do you run `python -m src.train` instead of `python src/train.py`?**
- **Say:** `-m` runs it as a module from the project root, so imports like `from src.data_loader import ...` work.
- **If they dig:** With `python src/train.py`, Python puts `src/` on the path instead of the project root, and `import src...` fails.

**B4. How does `predict_churn` avoid reloading the model every time?**
- **Say:** It caches the model and preprocessor in module-level variables and only loads them the first time.
- **In my project:** `_model` and `_preprocessor` globals in `load_artifacts()`.

**B5. Why is the customer input a dictionary?**
- **Say:** It maps each column name to a value. It's readable and converts directly into a 1-row DataFrame with `pd.DataFrame([customer])`, so the column names match what the preprocessor expects.

**B6. What is `pathlib.Path` doing?**
- **Say:** It builds file paths that work on any OS, relative to the project root, so the scripts work from any folder.
- **In my project:** `PROJECT_ROOT = Path(__file__).resolve().parent.parent`.

### C. Pandas

**C1. How did you load and inspect the data?**
- **Say:** `pd.read_csv`, then `.shape`, `.dtypes`, `.isna().sum()`, `.duplicated().sum()` and `value_counts(normalize=True)`.
- **In my project:** `describe_dataset()` in `data_loader.py`.

**C2. `isna()` showed 0 missing values. How did you find the missing data?**
- **Say:** `TotalCharges` had dtype object/string even though it should be numeric. That was the clue. 11 rows contained a blank space `" "`, which pandas doesn't count as NaN.
- **If they dig:** `pd.to_numeric(errors="coerce")` converts anything non-numeric to NaN, which exposed the 11. They all had tenure 0, meaning new customers who hadn't been billed yet.
- **In my project:** Always check `dtypes`, not just `isna()`.

**C3. What does `errors="coerce"` do?**
- **Say:** It converts invalid values to NaN instead of raising an error.

**C4. How did you compute churn rate by contract?**
- **Say:** `df.groupby("Contract")["Churn"].mean() * 100`. Because Churn is 0/1, the mean *is* the churn rate.
- **In my project:** Month-to-month 42.7%, two-year 2.8%.

**C5. What is `map` vs `apply`?**
- **Say:** `map` on a Series replaces values element by element using a dict or function. `apply` runs a function along rows/columns (or elements). For a simple lookup, `map` with a dict is simplest.
- **In my project:** `map({"Yes": 1, "No": 0})`.

**C6. Why `df.copy()` in `clean_data`?**
- **Say:** So I don't modify the caller's original DataFrame by accident.

**C7. What does `pd.cut` do?**
- **Say:** It bins a continuous column into intervals.
- **In my project:** Tenure bands 0–12, 13–24, 25–48, 49–72. First-year churn was 47.4%.

### D. NumPy

**D1. Where did you use NumPy?**
- **Say:** Converting the preprocessed matrices to `float32` arrays for TensorFlow, thresholding probabilities, `argmin` to find the best epoch, and building the class array for `compute_class_weight`.
- **In my project:** `y_pred = (y_prob >= 0.5).astype(int)` is a vectorised comparison over 1,409 predictions in one line.

**D2. What is vectorisation, and why does it matter?**
- **Say:** Applying an operation to a whole array at once in optimised C code instead of a Python loop. It's much faster.
- **If they dig:** StandardScaler computes `(X - mean) / std` for all rows using broadcasting.

**D3. What is broadcasting?**
- **Say:** NumPy automatically stretches a smaller array to match a bigger one. Subtracting a length-3 mean vector from a 4,507×3 matrix subtracts it from every row.

**D4. Why float32?**
- **Say:** TensorFlow uses float32 by default. It uses half the memory of float64, and the precision is enough for neural networks.

**D5. What does `.ravel()` do?**
- **Say:** It flattens an array. `model.predict` returns shape (n, 1), and `.ravel()` gives shape (n,), which is what sklearn metrics expect.

### E. Data preprocessing

**E1. Walk me through your preprocessing.**
- **Say:** Convert TotalCharges to numeric → drop customerID → encode target → stratified split → ColumnTransformer: numeric columns get median imputation + StandardScaler, categorical columns get most-frequent imputation + OneHotEncoder → fit on train, transform val/test.
- **In my project:** 19 raw columns become 46 numeric features.

**E2. How did you handle missing values? Why median?**
- **Say:** 11 missing TotalCharges, filled with the training median using SimpleImputer inside the pipeline.
- **If they dig:** The median is robust to skew (TotalCharges is right-skewed). Honestly, since all 11 have tenure 0, filling with **0** is arguably more logical. With 11 of 7,043 rows the choice has negligible impact. Keeping the imputer in the pipeline also protects the prediction function if a field is missing.
- **In my project:** `SimpleImputer(strategy="median")`.

**E3. Why one-hot encoding and not label encoding?**
- **Say:** The categories have no natural order. Label encoding would say "Electronic check = 0, Mailed check = 1, Bank transfer = 2", and the network would treat bank transfer as "bigger" than electronic check, which is meaningless.
- **If they dig:** One-hot gives each category its own 0/1 column. The downside is more columns (19 → 46), which is fine here. For very high-cardinality columns you'd consider other encodings.

**E4. Why `handle_unknown="ignore"`?**
- **Say:** If a new customer has a category not seen in training, the encoder outputs all zeros for that column instead of crashing.

**E5. Why did you scale features?**
- **Say:** Neural networks train with gradient descent, and features on very different scales (TotalCharges up to ~8,700 vs one-hot 0/1) make training unstable and slow, and let large-valued features dominate the early updates.
- **If they dig:** StandardScaler: z = (x − mean)/std, using the train mean/std. Tree models like Random Forest don't need scaling, but neural nets and Logistic Regression do.

**E6. What is data leakage, and how did you avoid it?**
- **Say:** Leakage means information from the test set (or the future) gets into training, so the test score is optimistic and fails in production.
- **If they dig:** If I fit the scaler on all 7,043 rows, the mean/std would include test customers. I split first, then `fit_transform` on train and only `transform` on val/test. The Pipeline makes this hard to get wrong.
- **In my project:** `preprocessor.fit_transform(X_train)` then `preprocessor.transform(X_val)` / `transform(X_test)`.

**E7. Why stratify the split?**
- **Say:** It keeps the same churn ratio (26.5%) in train, validation and test, so the test set represents reality. The output shows 0.265 in all three.

**E8. Why drop customerID?**
- **Say:** It's a unique identifier with no relationship to behaviour. A model could memorise it, which is useless for new customers.

**E9. You kept the 22 duplicate rows. Why?**
- **Say:** They're duplicates only when you ignore customerID. They are different customers who happen to have identical profiles, not data-entry errors.

### F. Scikit-learn

**F1. What is a Pipeline?**
- **Say:** A chain of steps (e.g. imputer → scaler) that you fit and transform as one object, so the same steps always run in the same order during training and prediction.

**F2. What is a ColumnTransformer?**
- **Say:** It applies different pipelines to different columns (numeric vs categorical) and joins the outputs side by side into one matrix.

**F3. `fit` vs `transform` vs `fit_transform`?**
- **Say:** `fit` learns parameters (median, mean, std, categories). `transform` applies them. `fit_transform` does both. Use it on train only.

**F4. How did you use Scikit-learn in a TensorFlow project?**
- **Say:** For the preprocessing pipeline, the train/test split, `compute_class_weight`, all evaluation metrics, and the Logistic Regression baseline. TensorFlow is only the neural network.

**F5. What does `compute_class_weight("balanced")` compute?**
- **Say:** weight = n_samples / (n_classes × count_of_class).
- **In my project:** 4,507 / (2 × 3,311) = 0.68 for stayers, and 4,507 / (2 × 1,196) = 1.88 for churners.

**F6. How did you save the preprocessor?**
- **Say:** With `joblib.dump` to `models/preprocessor.joblib`. The model is saved separately as `churn_dnn.keras`. At prediction time both are loaded, and they must match each other.

### G. TensorFlow / Keras

**G1. Why TensorFlow/Keras?**
- **Say:** Keras has a simple high-level API (`Sequential`, `compile`, `fit`), built-in callbacks like EarlyStopping, and easy model saving. It's widely used in industry. PyTorch would have worked just as well.

**G2. What does `model.compile` do?**
- **Say:** It configures training: the optimizer (Adam, lr 0.001), the loss (binary cross-entropy) and the metrics to track (accuracy, AUC, precision, recall).

**G3. What does `model.fit` do?**
- **Say:** It runs the training loop: for each epoch, it splits training data into batches of 32, does a forward pass, computes the loss, backpropagates and updates the weights, then evaluates on validation data.
- **In my project:** 141 batches per epoch (4,507 / 32).

**G4. How does EarlyStopping work?**
- **Say:** After each epoch it checks val_loss. If it hasn't improved for 10 epochs (patience), training stops, and `restore_best_weights=True` rolls back to the best epoch.
- **In my project:** Stopped at epoch 24, restored epoch 14.

**G5. How is class_weight applied in Keras?**
- **Say:** Each sample's loss is multiplied by its class weight, so errors on churners count about 2.8× more (1.88 / 0.68) when computing gradients.
- **If they dig:** It only affects the *training* loss. The validation loss Keras reports is unweighted.

**G6. How did you make results reproducible?**
- **Say:** `tf.keras.utils.set_random_seed(42)` seeds Python, NumPy and TensorFlow, and the splits use `random_state=42`. Running it twice gave identical numbers. Different library versions or hardware can still shift results slightly.

**G7. What's the output of `model.predict`?**
- **Say:** An (n, 1) array of probabilities between 0 and 1 from the sigmoid.

### H. Neural network / DNN concepts

**H1. What is a neuron?**
- **Say:** It computes a weighted sum of its inputs plus a bias, z = w·x + b, then applies an activation function.
- **In my project:** Each of the 32 neurons in layer 1 has 46 weights + 1 bias = 47 parameters.

**H2. What is a Dense layer?**
- **Say:** A fully connected layer: every neuron receives every output of the previous layer.
- **In my project:** 46 → 32 has 46 × 32 + 32 = 1,504 parameters.

**H3. What is ReLU, and why use it?**
- **Say:** ReLU(z) = max(0, z). It adds non-linearity, it's cheap, and it doesn't saturate for positive values, so gradients don't vanish the way they do with sigmoid in hidden layers.
- **If they dig:** Without a non-linear activation, stacked Dense layers collapse into a single linear model. The downside is "dying ReLU": neurons stuck outputting 0.

**H4. What is sigmoid, and why use it in the output?**
- **Say:** σ(z) = 1 / (1 + e^(−z)) squashes any number into (0, 1), so the single output can be read as the churn probability.
- **If they dig:** For binary classification: 1 sigmoid unit + binary cross-entropy. For multi-class: softmax + categorical cross-entropy.

**H5. What is dropout?**
- **Say:** During training it randomly switches off 30% of a layer's outputs on each step, so the network can't rely on specific neurons and generalises better. It's turned off at prediction time.
- **If they dig:** Keras scales the kept outputs by 1/(1−0.3) during training so the expected value stays the same at inference.

**H6. What happens if dropout is too high?**
- **Say:** Underfitting. With 0.8, most of the signal is thrown away on each step, the network can't learn, and both training and validation loss stay high.

**H7. What is binary cross-entropy?**
- **Say:** Loss = −[y·log(p) + (1−y)·log(1−p)]. It punishes confident wrong answers very heavily.
- **If they dig:** If a churner (y=1) gets p = 0.01, loss = −log(0.01) ≈ 4.6. If p = 0.9, loss ≈ 0.105. It's the negative log-likelihood of a Bernoulli outcome, and it pairs naturally with sigmoid.

**H8. What is an epoch? Batch size?**
- **Say:** An epoch is one full pass over the training data. Batch size is how many samples are processed before one weight update.
- **In my project:** Batch 32 means 141 weight updates per epoch. It trained for 24 epochs, so 24 × 141 ≈ 3,400 updates.

**H9. What is an optimizer? Why Adam?**
- **Say:** The optimizer decides how to update weights using the gradients. Adam adapts the step size for each parameter using running averages of the gradients (momentum) and the squared gradients. It works well with default settings.

**H10. What is the learning rate?**
- **Say:** The size of each weight-update step. Too high: loss jumps around or diverges. Too low: training is very slow or gets stuck.
- **In my project:** 0.001 (Adam's default).

**H11. What is backpropagation?**
- **Say:** The algorithm that computes the gradient of the loss with respect to every weight. It uses the chain rule, going backwards from the output layer to the input layer.

**H12. What is gradient descent?**
- **Say:** Repeatedly move each weight a small step in the direction that reduces the loss: w ← w − lr × ∂Loss/∂w. I used mini-batch gradient descent (batches of 32) through Adam.

**H13. Why is it called "deep"? Is 2 hidden layers really deep?**
- **Say:** Technically, "deep" means more than one hidden layer, so this is a small DNN / multi-layer perceptron. I kept it small on purpose: 2,049 parameters for 4,507 training rows. A bigger network would just overfit.

### I. EDA

**I1. Why did you do EDA before modelling?**
- **Say:** To find data problems (the TotalCharges text column, the imbalance), understand which features relate to churn, and have business insights to present even without a model.

**I2. What were your key EDA findings?**
- **Say:** Contract (month-to-month 42.7% vs two-year 2.8%), tenure (median 10 vs 38 months, 47.4% churn in year 1), charges (median monthly 79.65 vs 64.43), payment (electronic check 45.3%), and services (fiber 41.9%, no tech support 41.6%).

**I3. Explain the correlation heatmap.**
- **Say:** It shows Pearson correlations between the numeric columns and Churn. Tenure has −0.35 (longer tenure means less churn), MonthlyCharges +0.19, TotalCharges −0.20. TotalCharges and tenure correlate 0.83, so they're partly redundant.
- **If they dig:** Pearson only measures *linear* relationships, and correlation with a 0/1 target (point-biserial) is a rough indicator, not proof of cause. For categorical columns I compared churn rates per group instead.

**I4. Why do churners have LOWER total charges but HIGHER monthly charges?**
- **Say:** TotalCharges ≈ tenure × MonthlyCharges. Churners pay more per month but leave early, so their total is small. You'd reach the wrong conclusion looking at TotalCharges alone.

**I5. Why boxplots?**
- **Say:** They show median, spread (IQR) and outliers in one picture. They're good for comparing a numeric feature across the two churn groups.
- **In my project:** The tenure boxplot shows churners concentrated at low tenure, with a few long-tenure churners as outliers.

**I6. Does electronic check *cause* churn?**
- **Say:** No, it's an association. Electronic-check users are also more often month-to-month customers. The model can use it for prediction, but the business shouldn't assume that switching someone's payment method will keep them.

### J. Evaluation metrics

**J1. Why is accuracy misleading here?**
- **Say:** 73.5% of customers don't churn, so a useless model that always says "No" gets 73.5% accuracy and catches zero churners.
- **In my project:** My model's accuracy is 0.740, only slightly above 0.735, but its recall is 0.799 instead of 0.

**J2. Define precision and recall using your project.**
- **Say:** Precision is how many of the customers I flagged really churned: 299 / (299 + 291) = 0.507. Recall is how many of the real churners I caught: 299 / (299 + 75) = 0.799.

**J3. Explain your confusion matrix.**
- **Say:** TN 744 (stayers correctly left alone), FP 291 (stayers we'd wrongly give an offer), FN 75 (churners we missed, lost revenue), TP 299 (churners caught).

**J4. What's the business trade-off between FP and FN?**
- **Say:** A false positive costs a retention offer given to someone who would have stayed, e.g. a discount. A false negative costs a lost customer's future revenue, which is usually much bigger. So we lean towards recall. But if offers are expensive or the team can only call 100 people, precision matters more.
- **If they dig:** The right threshold depends on offer cost vs customer lifetime value and team capacity. No single metric is always "best".

**J5. What is F1?**
- **Say:** The harmonic mean of precision and recall, 2PR / (P + R). It's high only if both are high.
- **In my project:** 0.620.

**J6. What is ROC-AUC?**
- **Say:** The ROC curve plots true positive rate vs false positive rate across all thresholds. AUC is the area under it. 0.5 is random, 1.0 is perfect.
- **If they dig:** AUC = the probability that a randomly chosen churner gets a higher score than a randomly chosen non-churner. Mine is 0.845. It doesn't depend on the threshold, which makes it good for comparing models.

**J7. How does the threshold affect results?**
- **Say:** A lower threshold flags more customers: higher recall, lower precision.
- **In my project (validation set):** 0.3 gives recall 0.94 / precision 0.41. 0.5 gives 0.80 / 0.52. 0.7 gives 0.55 / 0.69.

**J8. Why not tune the threshold on the test set?**
- **Say:** That would be leakage. The test set must only be used once, for the final unbiased score. Threshold choices belong on validation data.

### K. ML theory

**K1. What is overfitting? How did you handle it?**
- **Say:** The model memorises training data and does worse on new data.
- **In my project:** I monitored training vs validation loss. When training loss kept dropping after epoch 14 but validation loss stopped improving, that was the start of overfitting. Dropout 0.3 and EarlyStopping with restored best weights (epoch 14) handled it. The model is also small: 2,049 parameters.

**K2. What is underfitting?**
- **Say:** The model is too simple, or too heavily regularised, to learn the pattern, so both training and validation scores are poor. Examples: dropout too high, too few neurons, stopping too early.

**K3. Bias vs variance?**
- **Say:** Bias is error from overly simple assumptions (underfitting). Variance is error from sensitivity to the particular training data (overfitting). Dropout and early stopping reduce variance.

**K4. Why train / validation / test, and not just train/test?**
- **Say:** Validation is used *during* development: early stopping and threshold decisions. Because I made decisions based on it, it's no longer unbiased. The test set is touched once at the end for an honest estimate.

**K5. What is regularisation?**
- **Say:** Any technique that discourages the model from fitting noise. Here: dropout and early stopping. L2 weight penalty is another option I didn't use.

**K6. What is cross-validation, and did you use it?**
- **Say:** Split the data into k folds, train k times with each fold used once as validation, then average. I used a single stratified split. Cross-validation would give a more reliable estimate and is listed as a future improvement.

### L. Project-specific troubleshooting

**L1. What happens if you run `astype(float)` directly on TotalCharges?**
- **Say:** It crashes with a ValueError because `" "` can't be converted to float. That's why I used `pd.to_numeric(errors="coerce")`.

**L2. Your validation loss is lower than training loss in early epochs. Is that a bug?**
- **Say:** No. Three reasons: dropout is active during training but not during validation; the training loss is multiplied by class weights while validation loss isn't; and the training loss is averaged over the epoch while the weights are still improving.

**L3. The model predicts "No churn" for everyone. What would you check?**
- **Say:** Whether class weights are actually passed to fit, whether features are scaled, whether the learning rate is reasonable, whether the label mapping is correct (Yes → 1), and whether the loss is decreasing at all.

**L4. The validation curve is noisy. Why?**
- **Say:** The validation set is small (1,127 rows, about 300 churners), so a few predictions flipping changes the metric. Dropout and mini-batches also add randomness. That's why patience = 10 instead of 1.

**L5. A new customer has a PaymentMethod never seen before. What happens?**
- **Say:** `handle_unknown="ignore"` encodes it as all zeros for that feature, and the prediction still runs.

**L6. Prediction fails after `git pull` on another machine. Why?**
- **Say:** The saved `preprocessor.joblib` is pickled with a specific scikit-learn version. Fix: run `python -m src.train` again to regenerate it, or pin versions.

**L7. Is the "93% probability" a true probability?**
- **Say:** Not exactly. Class weights push the predicted probabilities upward compared with the real 26.5% base rate, so they're good for *ranking* customers but not perfectly calibrated. For true probabilities I'd train without class weights and adjust the threshold instead, or calibrate on validation data.

### M. Resume cross-questioning

**M1. "Built a DNN using TensorFlow and Scikit-learn". What part is Scikit-learn?**
- **Say:** The DNN itself is Keras. Scikit-learn handles the preprocessing pipeline, splitting, class-weight computation, metrics and the Logistic Regression baseline.

**M2. "Analyze customer behavior". What behavior?**
- **Say:** Contract choice, tenure, services used, payment method, billing amounts. These are account and service attributes, not clickstream or usage logs. Be precise about this.

**M3. "End-to-end preprocessing pipeline". What makes it end-to-end?**
- **Say:** The same fitted pipeline goes from raw CSV columns to model-ready numbers, is saved to disk, and is reused unchanged on a raw customer dict at prediction time.

**M4. "Class imbalance". Which method? Why not SMOTE?**
- **Say:** Balanced class weights. They're simple, need no synthetic data, and are built into Keras `fit`. SMOTE creates synthetic minority samples, which adds complexity and has to be applied only to training data.
- **If they dig (real numbers, same seed and split):** Without class weights: accuracy **0.805**, recall **0.527**, precision 0.668, AUC 0.841. With class weights: accuracy 0.740, recall **0.799**, precision 0.507, AUC 0.845. So the weights didn't make the model "smarter" (AUC is nearly the same). They moved the decision boundary so it catches 102 more churners (299 vs 197), at the cost of more false alarms. Going by accuracy alone, you'd pick the *worse* model for retention.

**M5. "Relationships between tenure, pricing/charges, services and churn". Give numbers.**
- **Say:** See the key numbers table in section 2.

---

## 4. Strict interviewer cross-examination

**1. Why a DNN instead of Logistic Regression or Random Forest?**
Honest answer: *"I wanted to learn how to build and train a neural network end to end. I didn't assume it was better, so I trained a Logistic Regression baseline on the same features. The DNN got 0.845 AUC vs 0.842, basically the same. On small tabular data with mostly linear signals, simpler models are competitive. In production I'd probably pick Logistic Regression or gradient boosting for interpretability, unless the DNN clearly won."*

**2. Why TensorFlow?**
The Keras API is simple to learn, has built-in callbacks (EarlyStopping) and easy save/load, and is industry standard. PyTorch would work equally well. It's a tooling choice, not an accuracy choice.

**3. Why not Random Forest?**
Random Forest is a strong choice for tabular data. It needs no scaling, handles non-linearity and gives feature importances. I didn't use it because the project's focus was learning DNN training, but it's first on my future-improvement list. I wouldn't claim my DNN beats it without testing.

**4. Why did you scale the features?**
Gradient descent on unscaled inputs (TotalCharges up to ~8,700 next to 0/1 columns) is unstable and slow. StandardScaler puts the numeric features at mean 0, std 1, using the training statistics only.

**5. Why one-hot encoding?**
The categories are nominal (no order). Label encoding would invent a fake order. One-hot creates independent 0/1 columns: 19 raw features become 46.

**6. What happens if you fit the scaler before the train-test split?**
The scaler's mean/std would include the test customers. That's leakage: the test set is no longer truly unseen, and the reported score is slightly optimistic. With a single scaler on 7k rows the effect is small, but it's a wrong process, and with target-based steps (like target encoding or SMOTE before the split) the leak can be large.

**7. What is data leakage?**
Any information in training that wouldn't be available at real prediction time, like test statistics or future data. It makes offline scores look better than real-world performance.

**8. Why is accuracy insufficient for churn?**
Imbalance. 73.5% accuracy comes for free by predicting "No". My model's 74.0% accuracy says almost nothing. Recall 0.80 and AUC 0.845 show the real value.

**9. Explain precision vs recall using this project.**
Of the 590 customers flagged, 299 really churned: precision 0.51. Of the 374 real churners, 299 were caught: recall 0.80.

**10. What does your confusion matrix mean?**
TN 744, FP 291, FN 75, TP 299. The model trades 291 unnecessary offers to miss only 75 churners.

**11. Why sigmoid in the output layer?**
One neuron plus sigmoid outputs a value in (0, 1), which is read as P(churn). Softmax with 2 units would be equivalent but redundant.

**12. Why binary cross-entropy?**
It's the right loss for a Bernoulli target with a sigmoid output: the negative log-likelihood. It heavily penalises confident wrong predictions and gives strong gradients. MSE with sigmoid gives weak gradients when the prediction is badly wrong.

**13. Why ReLU?**
Non-linearity, cheap to compute, and no vanishing gradient for positive inputs. It's the standard default for hidden layers.

**14. Why dropout?**
Regularisation. It randomly zeros 30% of activations during training so the network can't memorise through specific neurons. It's off at inference.

**15. What happens if dropout is too high?**
Underfitting: training and validation loss both stay high because too much signal is dropped on every step.

**16. How did you handle class imbalance?**
`compute_class_weight("balanced")` gives {0: 0.68, 1: 1.88}, passed to `model.fit(class_weight=...)`. A missed churner costs about 2.8× more. Plus stratified splits and imbalance-aware metrics.

**17. Why train/validation/test?**
Train fits the weights. Validation drives early stopping and threshold decisions. Test is used once for the final unbiased estimate.

**18. What is an epoch?**
One full pass through all 4,507 training rows (141 batches of 32).

**19. What is batch size?**
The number of samples per gradient update. Smaller batches mean noisier but more frequent updates. Larger batches mean smoother updates but more memory and fewer steps per epoch. 32 is a common default.

**20. What is backpropagation?**
Using the chain rule to compute ∂Loss/∂weight for every weight, from the output layer backwards.

**21. What is gradient descent?**
Update the weights opposite to the gradient: w ← w − lr·∇L. Mini-batch version, via the Adam optimizer.

**22. What is the learning rate?**
The step size of each update. I used 0.001. Too high diverges or oscillates. Too low is slow.

**23. How do you know your model is actually learning?**
(a) Training and validation loss both fell from ~0.60/0.53 in the first epoch to ~0.49 at the best epoch. (b) Test AUC 0.845 is far above 0.5 (random). (c) Recall 0.80 vs 0 for the majority-class baseline. (d) Predictions make sense: a new month-to-month fiber customer on electronic check gets 93%, a 5-year two-year-contract customer gets 4%.

**24. How would you improve the model?**
Tune the threshold on validation using business costs. Cross-validation. Compare Random Forest / gradient boosting. Hyperparameter search. Explainability (permutation importance / SHAP). Add behavioural data (usage, complaints, support calls), which usually matters more than model choice.

**25. What if the model has 95% accuracy but poor recall?**
On imbalanced data, that means the model mostly predicts the majority class and misses churners, so it's useless for retention. I'd check the confusion matrix, add class weights or lower the threshold, and evaluate with recall / F1 / AUC.

**26. How would a telecom company actually use this model?**
Score every active customer monthly, rank them by churn probability, and have the retention team contact the top N their budget allows with a matching offer (e.g. a contract-upgrade discount for month-to-month fiber customers). Measure the effect with a control group (A/B test), monitor for data drift, and retrain periodically.

**27. What are the limitations of your project?**
One static public dataset, no time or usage data. The DNN only matches Logistic Regression and is less interpretable. One split, no cross-validation. Fixed 0.5 threshold. Probabilities aren't calibrated because of class weights. Correlation, not causation.

---

## 5. Resume defense: what you must know for each claim

| Claim | You must be able to… |
|---|---|
| **"Built a Deep Neural Network classifier"** | Draw the architecture (46→32→16→1), explain ReLU/sigmoid/dropout/BCE/Adam, calculate the 2,049 params, explain EarlyStopping (24 epochs, best 14), state the test metrics, and admit it only matches LR |
| **"End-to-end data preprocessing pipeline"** | Explain Pipeline vs ColumnTransformer, fit vs transform, why fitting on train only, how the same saved pipeline processes a raw dict in `predict_churn` |
| **"Handling missing values"** | Explain the hidden blanks in TotalCharges (11 rows, tenure 0), `to_numeric(errors="coerce")`, median imputation in the pipeline, and why 0 would also be defensible |
| **"Categorical encoding"** | One-hot vs label encoding, `handle_unknown="ignore"`, 16 categorical columns → 43 one-hot columns, the dummy-variable trap (matters for unregularised linear models, not for NNs) |
| **"Feature scaling"** | StandardScaler formula, why NNs need it and trees don't, fitted on train only |
| **"Class imbalance mitigation"** | 26.5% churn, the balanced class-weight formula and values (0.68 / 1.88), why accuracy fails, class weights vs SMOTE vs threshold tuning, that class weights reduce accuracy and inflate probabilities |
| **"Extensive EDA"** | Don't use "extensive". Know all 8 plots, the reason for each and the number behind each conclusion |
| **"Identified correlations between tenure, pricing and churn"** | Pearson values (tenure −0.35, Monthly +0.19, Total −0.20, tenure–Total 0.83), linear only, correlation ≠ causation, categorical relationships measured with churn rates |

---

## 6. 30 most likely interview questions (with one-line answers)

1. Explain your project. → 2-minute pitch.
2. What is churn? → A customer cancelling. Target Yes→1, No→0.
3. Which dataset, and what size? → IBM Telco, 7,043 × 21, 26.5% churn.
4. What was the target distribution? → 73.5% / 26.5%, imbalanced.
5. What data-quality issues did you find? → TotalCharges stored as text, 11 blanks for tenure-0 customers.
6. How did you handle missing values? → Coerce to NaN, then median imputer in the pipeline, fitted on train.
7. How did you encode categoricals? → OneHotEncoder, 19 → 46 features.
8. Why scale? → Gradient descent stability, features on the same scale.
9. What is data leakage? → Test/future info in training. Avoided by fitting on train only.
10. Why a Pipeline? → The same steps in the same order for train and prediction, no leakage.
11. Describe your model's architecture. → 46→32 ReLU→Drop→16 ReLU→Drop→1 sigmoid.
12. Why ReLU? → Non-linear, cheap, no vanishing gradient.
13. Why sigmoid? → Output is a probability in (0, 1).
14. Why binary cross-entropy? → Correct loss for a binary probability. Punishes confident wrong answers.
15. What does dropout do? → Randomly zeros 30% of activations while training. Regularisation.
16. What optimizer and learning rate? → Adam, 0.001.
17. What are epoch and batch size? → One full pass / 32 samples per update.
18. How did you prevent overfitting? → Dropout, EarlyStopping (restored epoch 14), small model.
19. How did you handle imbalance? → Balanced class weights {0: 0.68, 1: 1.88}.
20. Why not accuracy? → 73.5% for free by always predicting No.
21. What are your results? → Recall 0.80, precision 0.51, F1 0.62, AUC 0.845.
22. Explain your confusion matrix. → TN 744, FP 291, FN 75, TP 299.
23. Precision or recall: which is more important? → Depends on offer cost vs customer value. Usually recall-leaning.
24. What is ROC-AUC? → Ranking quality across all thresholds. 0.845.
25. Why a DNN over LR/RF? → Learning goal. I tested LR and it was equal. I'd try gradient boosting next.
26. What are the key EDA insights? → Contract, tenure, charges, payment method, tech support.
27. How does prediction work? → dict → DataFrame → saved preprocessor.transform → model.predict → threshold.
28. What are the limitations? → Static data, one split, uncalibrated probabilities, no causation.
29. What would you improve? → Threshold from costs, cross-validation, tree models, SHAP, behavioural data.
30. How would the business use it? → Monthly scoring, rank, contact top N, A/B test offers, retrain.

## 7. 20 rapid-fire questions

1. Rows? **7,043** · 2. Churn %? **26.5** · 3. Missing values? **11 TotalCharges** · 4. Features after encoding? **46** · 5. Hidden layers? **2 (32, 16)** · 6. Dropout rate? **0.3** · 7. Output activation? **Sigmoid** · 8. Loss? **Binary cross-entropy** · 9. Optimizer? **Adam** · 10. Learning rate? **0.001** · 11. Batch size? **32** · 12. Epochs run / best? **24 / 14** · 13. Patience? **10** · 14. Parameters? **2,049** · 15. Test recall? **0.80** · 16. Test AUC? **0.845** · 17. LR baseline AUC? **0.842** · 18. Class weight for churners? **1.88** · 19. Strongest single EDA signal? **Contract: 42.7% vs 2.8%** · 20. Threshold? **0.5**

## 8. 10 difficult follow-up questions

1. **Your accuracy is 74% and the dumb baseline is 73.5%. Is your model useless?** No. Accuracy hides it. My model catches 80% of churners where the baseline catches 0%, and AUC is 0.845 vs 0.5. The class weights intentionally trade accuracy for recall.
2. **The DNN equals Logistic Regression. Why keep the DNN?** For this data I wouldn't insist on it. LR is simpler and more interpretable. The DNN was a learning goal, and the baseline shows I didn't blindly assume complexity helps.
3. **Is 93% a real probability?** Not calibrated. Class weights inflate the scores, so they're good for ranking but not a literal probability. I'd calibrate on validation data or remove the weights and tune the threshold instead.
4. **You used validation for early stopping. Is your validation score biased?** Yes, slightly, because I made decisions based on it. That's exactly why I report the test set, which was used once.
5. **Imputing TotalCharges with the median for tenure-0 customers: is that right?** Not ideal. Their true total is ~0. It's 11 rows, so the impact is negligible, but filling with 0 or deriving it from tenure × MonthlyCharges is more logical.
6. **TotalCharges ≈ tenure × MonthlyCharges. Isn't that multicollinearity?** Yes (correlation 0.83 with tenure). It doesn't hurt prediction much for a NN, but it makes coefficient interpretation unreliable for linear models. I could drop it and test whether performance changes.
7. **Why patience 10 and not 3?** The validation loss is noisy with 1,127 samples. A small patience would stop on random bumps. `restore_best_weights` means extra epochs don't hurt.
8. **How would you choose the threshold properly?** Use validation data and a cost function: expected profit = TP × (saved value − offer cost) − FP × offer cost. Choose the threshold that maximises it, or pick top-N based on team capacity.
9. **What if churn behaviour changes next year?** Data drift. Monitor the distribution of inputs and the model's precision/recall on new labelled outcomes, and retrain periodically.
10. **How would you know which features drive a specific prediction?** Permutation importance globally, SHAP for individual customers. Neither is implemented yet, so it's in future improvements.

## 9. 10 "did you actually build it?" questions

1. **What dtype did TotalCharges load as, and why?** Text/object (string), because of 11 `" "` blanks.
2. **What do the rows with missing TotalCharges have in common?** tenure = 0, brand-new customers.
3. **How many columns after one-hot encoding?** 46.
4. **At which epoch did training stop, and which weights were kept?** Stopped at 24, restored 14.
5. **How many trainable parameters, and how do you calculate them?** 2,049 = (46×32+32) + (32×16+16) + (16+1).
6. **Walk me through `predict_churn` line by line.** Load cached artifacts → dict to 1-row DataFrame → check fields → coerce TotalCharges → `preprocessor.transform` (never fit) → float32 → `model.predict` → `[0, 0]` → compare with the threshold.
7. **Where is the validation set carved from?** 20% of the 80% training portion, stratified, which gives 4,507 / 1,127 / 1,409.
8. **What's the churn rate for month-to-month and for electronic check?** 42.7% and 45.3%.
9. **Why does validation loss sit below training loss early on?** Dropout and class weights are active only in training, and the training loss is an epoch average.
10. **What command trains the model, and where are outputs saved?** `python -m src.train` → `models/preprocessor.joblib`, `models/churn_dnn.keras`, and plots in `reports/figures/`.

---

## 10. One-page last-minute revision sheet

**Problem:** Predict telecom churn (binary) so retention can act early. **Data:** IBM Telco, 7,043 × 21, churn 26.5%.

**Cleaning:** TotalCharges text → `to_numeric(coerce)` → 11 NaN (tenure 0). Drop customerID. Churn Yes/No → 1/0. 22 profile duplicates kept.

**EDA:** Month-to-month 42.7% vs two-year 2.8% · tenure median 10 vs 38, year-1 churn 47.4% · monthly median 79.65 vs 64.43 · e-check 45.3% · fiber 41.9% · no tech support 41.6% · corr: tenure −0.35, monthly +0.19, total −0.20, tenure–total 0.83.

**Pipeline:** stratified split 64/16/20 → ColumnTransformer [numeric: median impute + StandardScaler | categorical: mode impute + OneHot(ignore)] → **fit on train only** → 46 features.

**Imbalance:** balanced class weights {0: 0.68, 1: 1.88} = n / (2 × class_count).

**Model:** 46 → Dense 32 ReLU → Dropout .3 → Dense 16 ReLU → Dropout .3 → Dense 1 Sigmoid · 2,049 params · BCE · Adam 0.001 · batch 32 · EarlyStopping(val_loss, patience 10, restore best) → stopped 24, best 14.

**Test results:** Acc 0.740 (baseline 0.735! and 0.805 without class weights, but recall only 0.527) · Precision 0.507 · **Recall 0.799** · F1 0.620 · **AUC 0.845** · CM: TN 744 FP 291 FN 75 TP 299 · LR baseline AUC 0.842 (≈ same).

**Trade-off:** FP = wasted offer. FN = lost customer (usually costlier). Threshold 0.3 → R .94 / P .41. 0.7 → R .55 / P .69 (validation). Choose by cost and capacity, on validation, never on test.

**Definitions in one line:**
- Neuron: w·x + b → activation
- ReLU: max(0, z)
- Sigmoid: 1/(1+e^−z)
- Dropout: randomly zero 30% during training
- BCE: −[y log p + (1−y) log(1−p)]
- Epoch: full pass
- Batch: samples per update
- Adam: adaptive per-parameter steps
- Learning rate: step size
- Backprop: chain-rule gradients
- Gradient descent: w −= lr·grad
- Overfitting: train ↑, val stalls/worsens
- Leakage: test info in training

**Honesty lines:** DNN ≈ LR on tabular data. Probabilities are not calibrated (class weights). Single split, no cross-validation. Correlation ≠ causation. Next steps: threshold from cost, cross-validation, gradient boosting, SHAP, behavioural data.
