## Energy Efficiency Prediction

### Documenting the results on our dataset

#### For Linear Regression Model

<ul>
<li>
Median value of the target: 18.25
</li>
<li>Percentage of 'high load' samples: 50.0 %</li>
<li>Mean Squared Error: 0.028401100966283708</li>
<li>
Accuracy: 0.9715988990337163
</li>
</ul>

#### For Logistic Regression Model
<ul>
<li>
Median value of the target: 21.5
</li>

<li>Percentage of 'high load' samples: 50.0 %</li>
<li>Evaluation Accuracy: 0.9675324675324676</li>
<li>Classification Report:</li>

                precision   recall  f1-score    support
       False       0.94      0.99      0.96        69
        True       0.99      0.95      0.97        85
    accuracy                           0.97       154
    macro avg      0.97     0.97       0.97       154
    weighted avg   0.97     0.97       0.97       154

</ul>

#### For Random Forest Model
<ul>
<li>
Median value of the target: 21.5
</li>

<li>Percentage of 'high load' samples: 50.0 %</li>
<li>Evaluation Accuracy: 1.0</li>
<li>Classification Report:</li>

                precision   recall  f1-score    support
       False       1.00      1.00      1.00        69
        True       1.00      1.00      1.00        85
    accuracy                           1.00       154
    macro avg      1.00     1.00       1.00       154
    weighted avg   1.00     1.00       1.00       154
</ul>